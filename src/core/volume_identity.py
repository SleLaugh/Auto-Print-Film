"""Strict identities of the decoded NIfTI volumes used by the pipeline.

Presentation metadata and the compressed representation are deliberately absent
from the identity. Geometry, pixel type, and every decoded voxel are included.
An optional local sidecar avoids decoding an unchanged file more than once;
passing no sidecar, or ``save=False``, never writes cache metadata.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk


_SCHEMA = "autoprintfilm-volume-identity-v1"
_SIDECAR_SCHEMA = "autoprintfilm-volume-identity-sidecar-v1"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class VolumeChangedError(RuntimeError):
    """The source changed while its header, identity, or sidecar was read."""


def _generation(path: Path) -> dict:
    source_stat = path.stat()
    return {
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "ctime_ns": source_stat.st_ctime_ns,
        "device": source_stat.st_dev,
        "inode": source_stat.st_ino,
    }


def _require_unchanged(path: Path, expected: dict) -> None:
    try:
        unchanged = _generation(path) == expected
    except OSError as exc:
        raise VolumeChangedError(f"Volume disappeared while reading: {path}") from exc
    if not unchanged:
        raise VolumeChangedError(f"Volume changed while reading: {path}")


def _geometry(size, pixel_type, components, spacing, origin, direction) -> dict:
    geometry = {
        "size": [int(value) for value in size],
        "pixel_type": int(pixel_type),
        "components": int(components),
        "spacing": [float(value) for value in spacing],
        "origin": [float(value) for value in origin],
        "direction": [float(value) for value in direction],
    }
    dimension = len(geometry["size"])
    if (
        dimension < 1
        or len(geometry["spacing"]) != dimension
        or len(geometry["origin"]) != dimension
        or len(geometry["direction"]) != dimension * dimension
        or any(value < 1 for value in geometry["size"])
        or geometry["components"] < 1
        or not all(
            math.isfinite(value)
            for key in ("spacing", "origin", "direction")
            for value in geometry[key]
        )
    ):
        raise ValueError("Invalid volume geometry")
    return geometry


def _header_summary(path: Path) -> dict:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return _geometry(
        reader.GetSize(), reader.GetPixelID(), reader.GetNumberOfComponents(),
        reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection(),
    )


def header_summary(path) -> dict:
    """Read exact geometry and pixel type without loading the voxel array.

    Equality is only a fast prefilter; callers must still compare full digests.
    A changing or disappearing source raises ``VolumeChangedError``.
    """
    source = Path(path)
    generation = _generation(source)
    summary = _header_summary(source)
    _require_unchanged(source, generation)
    return summary


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _identity_from_image(image) -> dict:
    geometry = _geometry(
        image.GetSize(), image.GetPixelID(), image.GetNumberOfComponentsPerPixel(),
        image.GetSpacing(), image.GetOrigin(), image.GetDirection(),
    )
    array = sitk.GetArrayViewFromImage(image)
    # No byte-order or traversal-order dependence; no rounding of voxel values.
    canonical = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    dtype = canonical.dtype.str
    exact_geometry = {
        key: [value.hex() for value in values]
        if key in {"spacing", "origin", "direction"} else values
        for key, values in geometry.items()
    }
    digest = hashlib.sha256()
    digest.update(_json_bytes({"schema": _SCHEMA, "geometry": exact_geometry, "dtype": dtype}))
    digest.update(b"\0")
    voxel_bytes = memoryview(canonical).cast("B")
    for offset in range(0, len(voxel_bytes), _HASH_CHUNK_BYTES):
        digest.update(voxel_bytes[offset:offset + _HASH_CHUNK_BYTES])
    return {
        "schema": _SCHEMA,
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "geometry": geometry,
        "dtype": dtype,
    }


def _valid_identity(identity) -> bool:
    try:
        if (
            not isinstance(identity, dict)
            or identity.get("schema") != _SCHEMA
            or identity.get("algorithm") != "sha256"
            or not isinstance(identity.get("digest"), str)
            or len(identity["digest"]) != 64
            or any(character not in "0123456789abcdef" for character in identity["digest"])
            or np.dtype(identity["dtype"]).str != identity["dtype"]
        ):
            return False
        geometry = identity["geometry"]
        return geometry == _geometry(
            geometry["size"], geometry["pixel_type"], geometry["components"],
            geometry["spacing"], geometry["origin"], geometry["direction"],
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _read_sidecar(path: Path, generation: dict):
    try:
        with path.open("r", encoding="utf-8") as stream:
            envelope = json.load(stream)
        if not isinstance(envelope, dict):
            return None
        payload = {key: envelope[key] for key in ("schema", "generation", "identity")}
        if (
            payload["schema"] != _SIDECAR_SCHEMA
            or payload["generation"] != generation
            or envelope.get("integrity_sha256") != hashlib.sha256(_json_bytes(payload)).hexdigest()
            or not _valid_identity(payload["identity"])
        ):
            return None
        return payload["identity"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, OverflowError):
        return None


def _write_sidecar(path: Path, source: Path, generation: dict, identity: dict) -> None:
    payload = {"schema": _SIDECAR_SCHEMA, "generation": generation, "identity": identity}
    envelope = dict(payload, integrity_sha256=hashlib.sha256(_json_bytes(payload)).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(envelope))
            stream.flush()
            os.fsync(stream.fileno())
        _require_unchanged(source, generation)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_volume_identity(path, metadata_path=None, *, save: bool = True) -> dict:
    """Return a strict decoded-volume SHA256, geometry, and canonical dtype.

    An explicit ``metadata_path`` enables stat-validated sidecar reuse and, by
    default, atomic sidecar publication. ``save=False`` may read an existing
    sidecar but never changes it, including when it is absent or invalid.
    Sidecars are local generation hints, not portable or authoritative image
    identities. Metadata I/O errors during an opted-in write propagate.
    """
    source = Path(path)
    generation = _generation(source)
    sidecar = Path(metadata_path) if metadata_path is not None else None
    if sidecar is not None and sidecar.resolve() == source.resolve():
        raise ValueError("Volume identity metadata must not replace the source volume")
    if sidecar is not None:
        identity = _read_sidecar(sidecar, generation)
        if identity is not None:
            _require_unchanged(source, generation)
            return identity
    image = sitk.ReadImage(str(source))
    identity = _identity_from_image(image)
    _require_unchanged(source, generation)
    if sidecar is not None and save:
        _write_sidecar(sidecar, source, generation, identity)
        _require_unchanged(source, generation)
    return identity
