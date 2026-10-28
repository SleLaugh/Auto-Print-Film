"""Strict volume identities and generation-bound, optional local sidecars."""

import gzip
import json
import os

import numpy as np
import pytest
import SimpleITK as sitk

from src.core import volume_identity as identity_module
from src.core.volume_identity import (
    VolumeChangedError,
    get_volume_identity,
    header_summary,
)


def _image(dtype=np.int16):
    image = sitk.GetImageFromArray(np.arange(60, dtype=dtype).reshape(3, 4, 5))
    image.SetSpacing((0.75, 0.875, 2.0))
    image.SetOrigin((12.0, -4.0, 9.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _write(tmp_path, name="raw.nii.gz", image=None):
    path = tmp_path / name
    sitk.WriteImage(_image() if image is None else image, str(path))
    return path


def _bump_mtime(path):
    previous = path.stat()
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))


def test_same_decoded_volume_ignores_compression_and_presentation_metadata(tmp_path):
    source = _write(tmp_path, "raw.nii")
    payload = bytearray(source.read_bytes())
    # NIfTI-1 descrip stores presentation/description text, not image geometry.
    payload[148:228] = b"Different display window metadata".ljust(80, b"\0")
    compressed = tmp_path / "other.nii.gz"
    compressed.write_bytes(gzip.compress(bytes(payload), compresslevel=1, mtime=12))

    first = get_volume_identity(source)
    second = get_volume_identity(compressed)

    assert first == second
    assert len(first["digest"]) == 64
    assert first["dtype"] == "<i2"
    assert header_summary(source) == first["geometry"]
    assert header_summary(compressed) == first["geometry"]


@pytest.mark.parametrize("change", ["voxel", "spacing", "origin", "direction", "type"])
def test_identity_rejects_different_voxels_geometry_and_pixel_type(tmp_path, change):
    source = _write(tmp_path)
    image = _image()
    if change == "voxel":
        image[2, 1, 1] += 1
    elif change == "spacing":
        image.SetSpacing((0.75, 0.875, 2.00001))
    elif change == "origin":
        image.SetOrigin((12.0, -4.0, 9.00001))
    elif change == "direction":
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    else:
        image = sitk.Cast(image, sitk.sitkFloat32)
    changed = _write(tmp_path, "changed.nii.gz", image)
    assert get_volume_identity(source)["digest"] != get_volume_identity(changed)["digest"]


def test_default_identity_does_not_create_sidecar(tmp_path):
    source = _write(tmp_path)
    before = set(tmp_path.iterdir())
    get_volume_identity(source)
    assert set(tmp_path.iterdir()) == before


def test_valid_generation_reuses_sidecar_without_decoding(tmp_path, monkeypatch):
    source = _write(tmp_path)
    sidecar = tmp_path / "volume_identity.json"
    expected = get_volume_identity(source, sidecar)
    before = sidecar.stat()

    def no_decode(*args, **kwargs):
        pytest.fail("unchanged generation must not decode the image again")

    monkeypatch.setattr(sitk, "ReadImage", no_decode)
    assert get_volume_identity(source, sidecar) == expected
    assert sidecar.stat() == before


def test_replacing_source_invalidates_even_with_preserved_size_and_mtime(tmp_path):
    source = _write(tmp_path, "raw.nii")
    sidecar = tmp_path / "volume_identity.json"
    previous = get_volume_identity(source, sidecar)
    previous_stat = source.stat()
    changed_image = _image()
    changed_image[0, 0, 0] = 123
    replacement = _write(tmp_path, "replacement.nii", changed_image)
    assert replacement.stat().st_size == previous_stat.st_size
    os.utime(replacement, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    os.replace(replacement, source)

    current = get_volume_identity(source, sidecar)

    assert current["digest"] != previous["digest"]
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    assert envelope["generation"]["inode"] == source.stat().st_ino


def test_in_place_source_mutation_invalidates_sidecar(tmp_path):
    source = _write(tmp_path, "raw.nii")
    sidecar = tmp_path / "volume_identity.json"
    previous = get_volume_identity(source, sidecar)
    previous_stat = source.stat()
    with source.open("r+b") as stream:
        stream.seek(-2, 2)
        stream.write(b"\xff\x00")
    os.utime(source, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    assert get_volume_identity(source, sidecar)["digest"] != previous["digest"]


@pytest.mark.parametrize("contents", ["not json", "[]", '{"schema":"old"}'])
def test_corrupt_sidecar_is_recomputed(tmp_path, contents):
    source = _write(tmp_path)
    expected = get_volume_identity(source)
    sidecar = tmp_path / "volume_identity.json"
    sidecar.write_text(contents)
    assert get_volume_identity(source, sidecar) == expected
    assert json.loads(sidecar.read_text(encoding="utf-8"))["identity"] == expected


def test_sidecar_integrity_detects_valid_json_corruption(tmp_path):
    source = _write(tmp_path)
    sidecar = tmp_path / "volume_identity.json"
    expected = get_volume_identity(source, sidecar)
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["identity"]["digest"] = "a" * 64
    sidecar.write_text(json.dumps(envelope))
    assert get_volume_identity(source, sidecar) == expected


@pytest.mark.parametrize("sidecar_state", ["missing", "invalid", "stale"])
def test_read_only_lookup_never_rewrites_sidecar(tmp_path, sidecar_state):
    source = _write(tmp_path)
    sidecar = tmp_path / "volume_identity.json"
    if sidecar_state == "invalid":
        sidecar.write_text("invalid")
    elif sidecar_state == "stale":
        get_volume_identity(source, sidecar)
        image = _image()
        image[0, 0, 0] = 98
        sitk.WriteImage(image, str(source))
    before = sidecar.read_bytes() if sidecar.exists() else None

    assert get_volume_identity(source, sidecar, save=False) == get_volume_identity(source)
    assert (sidecar.read_bytes() if sidecar.exists() else None) == before


def test_source_change_during_decode_is_rejected_without_publication(tmp_path, monkeypatch):
    source = _write(tmp_path)
    sidecar = tmp_path / "volume_identity.json"
    read_image = sitk.ReadImage

    def changing_read(path):
        image = read_image(path)
        _bump_mtime(source)
        return image

    monkeypatch.setattr(sitk, "ReadImage", changing_read)
    with pytest.raises(VolumeChangedError):
        get_volume_identity(source, sidecar)
    assert not sidecar.exists()


def test_source_change_during_header_read_is_rejected(tmp_path, monkeypatch):
    source = _write(tmp_path)
    read_header = identity_module._header_summary

    def changing_header(path):
        summary = read_header(path)
        _bump_mtime(source)
        return summary

    monkeypatch.setattr(identity_module, "_header_summary", changing_header)
    with pytest.raises(VolumeChangedError):
        header_summary(source)


def test_source_change_during_sidecar_lookup_is_rejected(tmp_path, monkeypatch):
    source = _write(tmp_path)
    sidecar = tmp_path / "volume_identity.json"
    get_volume_identity(source, sidecar)
    read_sidecar = identity_module._read_sidecar

    def changing_lookup(path, generation):
        result = read_sidecar(path, generation)
        _bump_mtime(source)
        return result

    monkeypatch.setattr(identity_module, "_read_sidecar", changing_lookup)
    with pytest.raises(VolumeChangedError):
        get_volume_identity(source, sidecar)


def test_atomic_write_failure_preserves_old_metadata_and_removes_temporary(tmp_path, monkeypatch):
    source = _write(tmp_path)
    sidecar = tmp_path / "volume_identity.json"
    sidecar.write_text("old metadata")

    def failing_replace(*args):
        raise OSError("test atomic publication failure")

    monkeypatch.setattr(identity_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="test atomic publication failure"):
        get_volume_identity(source, sidecar)
    assert sidecar.read_text(encoding="utf-8") == "old metadata"
    assert not list(tmp_path.glob(".volume_identity.json.*.tmp"))


def test_hash_normalizes_array_byte_order_and_traversal(tmp_path, monkeypatch):
    source = _write(tmp_path)
    expected = get_volume_identity(source)
    get_array = sitk.GetArrayViewFromImage

    def other_layout(image):
        return np.asfortranarray(get_array(image).astype(">i2"))

    monkeypatch.setattr(sitk, "GetArrayViewFromImage", other_layout)
    assert get_volume_identity(source) == expected


def test_metadata_cannot_replace_source_volume(tmp_path):
    source = _write(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must not replace"):
        get_volume_identity(source, source)
    assert source.read_bytes() == before
