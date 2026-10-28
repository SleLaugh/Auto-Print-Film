"""Lossless cache of final, unwindowed reconstructed images and their geometry.

The index is private to a series-reuse bundle. Its owner publishes the bundle
only after the complete task succeeds; these files never act as ready markers.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import SimpleITK as sitk


_SCHEMA = "reconstruction_hu_v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported HU cache value: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("HU cache file escapes its bundle")
    return path


def initialize_hu_cache(path: Path, plans: list[dict[str, Any]]) -> None:
    """Declare the complete plan set before any images are captured."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    expected = [str(plan["plan_id"]) for plan in plans]
    if len(set(expected)) != len(expected):
        raise ValueError("HU cache plan IDs must be unique")
    _write_json(path / "index.json", {
        "schema": _SCHEMA, "expected_plan_ids": expected, "plans": {},
    })


def _save_array(path: Path, array: np.ndarray) -> dict[str, str]:
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ValueError("HU cache arrays must contain real numbers")
    # np.savez is intentionally uncompressed: preserve exact dtype/values and
    # avoid spending CPU compressing the already small reconstructed stacks.
    with path.open("wb") as stream:
        np.savez(stream, array=array)
    return {"file": path.name, "sha256": _digest(path)}


def _save_image(path: Path, image: sitk.Image) -> dict[str, Any]:
    result = _save_array(path, sitk.GetArrayViewFromImage(image))
    result.update({
        "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "metadata": {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()},
    })
    return result


def capture_plan(
    path: Path,
    plan: dict[str, Any],
    sampled_slices: list[tuple[dict[str, Any], sitk.Image]],
    localizer: dict[str, Any] | None,
) -> None:
    """Capture final HU/framing and the unwindowed localizer for one plan."""
    path = Path(path)
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    plan_id = str(plan["plan_id"])
    if plan_id not in index["expected_plan_ids"]:
        raise ValueError("Undeclared HU cache plan")
    prefix = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()
    slices = [
        {"planned_slice": planned_slice,
         "image": _save_image(path / f"{prefix}-slice-{ordinal}.npz", image)}
        for ordinal, (planned_slice, image) in enumerate(sampled_slices)
    ]
    localizer_record = None
    if localizer is not None:
        localizer_record = {
            "background_image": _save_image(
                path / f"{prefix}-localizer.npz", localizer["background_image"],
            ),
            "line_alpha": _save_array(
                path / f"{prefix}-alpha.npz", np.asarray(localizer["line_alpha"]),
            ),
            "legacy_line_value": float(localizer["legacy_line_value"]),
            "background_kind": str(localizer["background_kind"]),
        }
    record_path = path / f"{prefix}.json"
    _write_json(record_path, {"plan": plan, "slices": slices, "localizer": localizer_record})
    index["plans"][plan_id] = {
        "file": record_path.name, "sha256": _digest(record_path),
    }
    _write_json(path / "index.json", index)


def _load_array(root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    path = _file(root, descriptor["file"])
    if _digest(path) != descriptor["sha256"]:
        raise ValueError("HU cache checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"array"}:
            raise ValueError("Unexpected HU cache array fields")
        array = archive["array"]
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ValueError("Invalid HU cache array dtype")
    return array


def _load_image(root: Path, descriptor: dict[str, Any]) -> sitk.Image:
    array = _load_array(root, descriptor)
    size = descriptor["size"]
    dimension = len(size)
    if dimension not in (2, 3) or tuple(reversed(size)) != array.shape:
        raise ValueError("HU cache image shape mismatch")
    spacing = np.asarray(descriptor["spacing"], dtype=float)
    origin = np.asarray(descriptor["origin"], dtype=float)
    direction = np.asarray(descriptor["direction"], dtype=float)
    if (spacing.shape != (dimension,) or origin.shape != (dimension,)
            or direction.shape != (dimension * dimension,)
            or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction))):
        raise ValueError("Invalid HU cache image geometry")
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(origin))
    image.SetDirection(tuple(direction))
    for key, value in descriptor["metadata"].items():
        image.SetMetaData(key, value)
    return image


def iter_cached_plans(path: Path) -> Iterator[dict[str, Any]]:
    """Load one plan at a time so several planes do not accumulate in memory."""
    path = Path(path)
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    expected = index["expected_plan_ids"]
    if (index.get("schema") != _SCHEMA or not isinstance(expected, list)
            or len(set(expected)) != len(expected)
            or set(index["plans"]) != set(expected)):
        raise ValueError("Incomplete HU cache plan set")
    for plan_id in expected:
        descriptor = index["plans"][plan_id]
        record_path = _file(path, descriptor["file"])
        if _digest(record_path) != descriptor["sha256"]:
            raise ValueError("HU cache plan checksum mismatch")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        plan = record["plan"]
        if str(plan["plan_id"]) != plan_id or len(record["slices"]) != len(plan["slices"]):
            raise ValueError("HU cache plan/slice count mismatch")
        if [item["planned_slice"]["index"] for item in record["slices"]] != [
            item["index"] for item in plan["slices"]
        ]:
            raise ValueError("HU cache slice order mismatch")
        sampled_slices = [
            (item["planned_slice"], _load_image(path, item["image"]))
            for item in record["slices"]
        ]
        localizer = record["localizer"]
        if localizer is not None:
            localizer = {
                **localizer,
                "background_image": _load_image(path, localizer["background_image"]),
                "line_alpha": _load_array(path, localizer["line_alpha"]),
            }
            alpha = localizer["line_alpha"]
            background = localizer["background_image"]
            if (background.GetDimension() != 2
                    or alpha.shape != tuple(reversed(background.GetSize()))
                    or not np.all(np.isfinite(alpha))
                    or np.any(alpha < 0) or np.any(alpha > 1)
                    or not np.isfinite(localizer["legacy_line_value"])
                    or not isinstance(localizer["background_kind"], str)):
                raise ValueError("Invalid HU cache localizer coverage")
            for tag in ("0020|0032", "0020|0037"):
                if not background.HasMetaDataKey(tag) or not background.GetMetaData(tag).strip():
                    raise ValueError("Missing HU cache localizer geometry")
        yield {"plan": plan, "sampled_slices": sampled_slices, "localizer": localizer}


def validate_hu_cache(path: Path) -> bool:
    """Check every record and image before a bundle is accepted as a cache hit."""
    try:
        for _ in iter_cached_plans(path):
            pass
        return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, zipfile.BadZipFile):
        return False
