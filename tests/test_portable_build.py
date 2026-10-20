"""Executable builds publish current JSON, with explicit reuse of earlier edits."""

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from unittest.mock import Mock

import pytest

from scripts import build_portable


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(autouse=True)
def forbid_downloads(monkeypatch):
    monkeypatch.setattr(
        build_portable.urllib.request, "urlopen",
        Mock(side_effect=AssertionError("Tests must not download a runtime")),
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "development source"
    for name in ("main.py", "README.md", "docs/production.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source file: {name}\n", encoding="utf-8")
    write_json(root / "src/config/nnuNet_config.json", {
        "_configuration": "3d_fullres",
        "nnUNet_raw": "/configured-models/nnUNet_raw",
        "nnUNet_preprocessed": "/configured-models/nnUNet_preprocessed",
        "nnUNet_results": "/configured-models/nnUNet_results",
    })
    write_json(root / "src/config/logger_config.json", {"base_dir": "OperatorLog"})
    write_json(root / "src/config/slice_part_config.json", {"revision": "source-default"})
    write_json(root / "src/config/window_presets.json", [
        {"name": "脑窗", "window_width": 80, "window_center": 40},
    ])
    write_json(root / "build-config.json", {
        "python_version": "3.11.17",
        "python_url": "https://example.invalid/python.tar.gz",
        "python_sha256": "a" * 64,
    })
    (root / "constraints-build.txt").write_text("stub==1.0\n", encoding="utf-8")
    (root / "requirements.txt").write_text("stub==1.0\n", encoding="utf-8")
    (root / "requirements-build.txt").write_text("pyinstaller==6.16.0\n", encoding="utf-8")
    (root / "AutoPrintFilm.spec").write_text("# executable build specification", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts/frozen_entry.py").write_text("# frozen entrypoint", encoding="utf-8")
    (root / "src/core").mkdir(parents=True)
    (root / "src/core/pipeline.py").write_text("# application source", encoding="utf-8")
    (root / "dev").write_text("# developer command", encoding="utf-8")
    monkeypatch.setattr(
        build_portable, "_git_metadata",
        lambda _: {"head": "test-head", "dirty": True, "available": True},
    )
    monkeypatch.setattr(build_portable.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_portable.platform, "machine", lambda: "x86_64")
    return root


@pytest.fixture
def simulated_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime cache/python"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python3").write_text("# interpreter stand-in", encoding="utf-8")
    ensure = Mock(return_value=runtime)

    def freeze(source, build_runtime, temporary):
        assert build_runtime == runtime
        assert (source / "AutoPrintFilm.spec").is_file()
        application = temporary / "pyinstaller-output/AutoPrintFilm"
        (application / "_internal").mkdir(parents=True)
        executable = application / "AutoPrintFilm"
        executable.write_bytes(b"executable stand-in")
        executable.chmod(0o755)
        (application / "_internal/dep.so").write_bytes(b"bundled dependency")
        (application / "requirements.lock.txt").write_text("stub==1.0\n", encoding="utf-8")
        return application

    def validate(application):
        executable = application / "AutoPrintFilm"
        assert executable.is_file() and os.access(executable, os.X_OK)
        assert (application / "_internal/dep.so").is_file()
        assert not (application / "python").exists()
        assert not (application / "main.py").exists()

    validator = Mock(side_effect=validate)
    freezer = Mock(side_effect=freeze)
    monkeypatch.setattr(build_portable, "ensure_runtime", ensure)
    monkeypatch.setattr(build_portable, "validate_application", validator)
    monkeypatch.setattr(build_portable, "freeze_application", freezer)
    monkeypatch.setattr(
        build_portable.subprocess, "run",
        Mock(side_effect=AssertionError("Tests must not execute dependency installation")),
    )
    return ensure, validator, freezer


def archived_json(archive, relative):
    with tarfile.open(archive, "r:gz") as bundle:
        with bundle.extractfile(f"AutoPrintFilm/{relative}") as stream:
            return json.load(stream)


def artifact_snapshot(dist):
    return {
        path.relative_to(dist).as_posix(): path.read_bytes()
        for path in dist.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("reset_config", [False, True])
def test_rebuild_can_explicitly_preserve_or_replace_edited_json(
    source, simulated_runtime, reset_config,
):
    output = build_portable.build_portable(source)
    source_configuration = {"revision": "new source-default", "窗位": "骨窗"}
    operator_configuration = {"revision": "operator-edited", "窗位": "自定义"}
    write_json(source / "src/config/slice_part_config.json", source_configuration)
    write_json(output / "src/config/slice_part_config.json", operator_configuration)
    write_json(output / "src/config/operator-extra.json", {"enabled": True})

    rebuilt = build_portable.build_portable(source, reset_config=reset_config)

    expected = source_configuration if reset_config else operator_configuration
    relative = "src/config/slice_part_config.json"
    assert rebuilt == output
    assert json.loads((output / relative).read_text(encoding="utf-8")) == expected
    assert archived_json(source / "dist" / build_portable.ARCHIVE_NAME, relative) == expected
    assert (output / "src/config/operator-extra.json").exists() is not reset_config
    assert json.loads((source / relative).read_text(encoding="utf-8")) == source_configuration
    manifest = json.loads((output / "RELEASE.json").read_text(encoding="utf-8"))
    assert manifest["files"][relative] == hashlib.sha256((output / relative).read_bytes()).hexdigest()


def test_default_rebuild_publishes_all_current_source_json_in_directory_and_archive(
    source, simulated_runtime,
):
    output = build_portable.build_portable(source)
    # Model a previous release with obsolete names and operator-only settings.
    write_json(output / "src/config/slice_part_config.json", {"部位": "头骨", "标签": "skull"})
    write_json(output / "src/config/operator-extra.json", {"enabled": True})
    for path in (source / "src/config").glob("*.json"):
        updated = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(updated, dict):
            updated["revision"] = "new source"
        else:
            updated[0]["window_center"] = 45
        write_json(path, updated)
    write_json(source / "src/config/slice_part_config.json", {
        "部位": "颅脑", "标签": "brain", "start_mm": 10,
    })
    # Future source configurations must also be published without a fixed list.
    write_json(source / "src/config/new-setting.json", {"enabled": True})

    build_portable.build_portable(source)

    assert not (output / "src/config/operator-extra.json").exists()
    manifest = json.loads((output / "RELEASE.json").read_text(encoding="utf-8"))
    archive = source / "dist" / build_portable.ARCHIVE_NAME
    for path in (source / "src/config").glob("*.json"):
        relative = f"src/config/{path.name}"
        assert (output / relative).read_bytes() == path.read_bytes()
        assert archived_json(archive, relative) == json.loads(path.read_text(encoding="utf-8"))
        assert manifest["files"][relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(archive, "r:gz") as bundle:
        assert "AutoPrintFilm/src/config/operator-extra.json" not in bundle.getnames()


def test_keep_config_still_publishes_new_source_configuration(source, simulated_runtime):
    output = build_portable.build_portable(source)
    write_json(source / "src/config/new-setting.json", {"enabled": True})
    write_json(output / "src/config/logger_config.json", {"base_dir": "EditedLogs"})

    build_portable.build_portable(source, reset_config=False)

    assert archived_json(source / "dist" / build_portable.ARCHIVE_NAME,
                         "src/config/new-setting.json") == {"enabled": True}
    assert json.loads((output / "src/config/logger_config.json").read_text()) == {
        "base_dir": "EditedLogs",
    }


def test_default_rebuild_replaces_invalid_old_json_without_reading_it(source, simulated_runtime):
    output = build_portable.build_portable(source)
    (output / "src/config/slice_part_config.json").write_text("broken old JSON", encoding="utf-8")

    build_portable.build_portable(source)

    assert (output / "src/config/slice_part_config.json").read_bytes() == (
        source / "src/config/slice_part_config.json"
    ).read_bytes()


@pytest.mark.parametrize("options, reset_config", [
    ([], True), (["--reset-config"], True), (["--keep-config"], False),
])
def test_cli_config_policy_defaults_to_current_source(monkeypatch, options, reset_config):
    build = Mock()
    monkeypatch.setattr(build_portable, "build_portable", build)

    assert build_portable.main(["build", *options]) == 0

    build.assert_called_once_with(refresh=False, reset_config=reset_config)


def test_cli_rejects_conflicting_config_options(monkeypatch):
    build = Mock()
    monkeypatch.setattr(build_portable, "build_portable", build)

    with pytest.raises(SystemExit) as error:
        build_portable.main(["build", "--keep-config", "--reset-config"])

    assert error.value.code == 2
    build.assert_not_called()


def test_invalid_operator_json_aborts_without_replacing_previous_artifacts(source, simulated_runtime):
    output = build_portable.build_portable(source)
    (output / "src/config/slice_part_config.json").write_text("broken operator JSON", encoding="utf-8")
    before = artifact_snapshot(source / "dist")

    with pytest.raises(ValueError):
        build_portable.build_portable(source, reset_config=False)

    assert artifact_snapshot(source / "dist") == before
    assert not list((source / "dist").glob(".autoprintfilm-build-*"))


def test_build_constraints_stay_in_the_source_project(source, simulated_runtime):
    output = build_portable.build_portable(source)

    assert not (output / "constraints-build.txt").exists()
    assert (source / "constraints-build.txt").read_text(encoding="utf-8") == "stub==1.0\n"
    with tarfile.open(source / "dist" / build_portable.ARCHIVE_NAME, "r:gz") as bundle:
        assert "AutoPrintFilm/constraints-build.txt" not in bundle.getnames()
    assert json.loads((output / "PORTABLE.json").read_text(encoding="utf-8"))["build_tool"] == (
        "AutoPrintFilm dev build"
    )


def test_application_bundle_declares_external_inference_and_excludes_legacy_install_tools(
    source, simulated_runtime,
):
    output = build_portable.build_portable(source)

    marker = json.loads((output / "PORTABLE.json").read_text(encoding="utf-8"))
    assert marker["inference_environment"] == "external"
    assert marker["packaging"] == "pyinstaller-onedir"
    assert marker["executable"] == "AutoPrintFilm"
    assert set(marker["settings"]) == {"python_version", "python_url", "python_sha256"}
    assert not (output / "scripts").exists()
    assert not (output / "main.py").exists()
    assert not (output / "python").exists()
    assert not list(output.rglob("*.py"))
    executable = output / marker["executable"]
    assert executable.is_file() and os.access(executable, os.X_OK)
    assert (output / "_internal/dep.so").is_file()
    manifest = json.loads((output / "RELEASE.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"RELEASE.json", "PORTABLE.json"}
    }
    for relative, digest in manifest["files"].items():
        assert digest == hashlib.sha256((output / relative).read_bytes()).hexdigest()
    with tarfile.open(source / "dist" / build_portable.ARCHIVE_NAME, "r:gz") as bundle:
        assert not any(name.startswith("AutoPrintFilm/scripts/") for name in bundle.getnames())
        assert not any(name.endswith(".py") for name in bundle.getnames())


@pytest.mark.parametrize("marker", [None, {"build_tool": "another build tool"}])
def test_unknown_output_is_rejected_without_installing_or_changing_it(source, simulated_runtime, marker):
    output = source / "dist/AutoPrintFilm"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("operator-owned directory", encoding="utf-8")
    if marker is not None:
        write_json(output / "PORTABLE.json", marker)
    before = artifact_snapshot(source / "dist")

    with pytest.raises(ValueError):
        build_portable.build_portable(source)

    ensure, validator, freezer = simulated_runtime
    ensure.assert_not_called()
    freezer.assert_not_called()
    validator.assert_not_called()
    assert artifact_snapshot(source / "dist") == before


@pytest.mark.parametrize("relative", ["dist", ".build-cache", "dist/AutoPrintFilm"])
def test_symlink_build_directories_do_not_redirect_builds(source, simulated_runtime, tmp_path, relative):
    target = tmp_path / "operator-owned directory"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("must not change", encoding="utf-8")
    link = source / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError):
        build_portable.build_portable(source)

    ensure, validator, freezer = simulated_runtime
    ensure.assert_not_called()
    freezer.assert_not_called()
    validator.assert_not_called()
    assert marker.read_text(encoding="utf-8") == "must not change"
    assert set(target.iterdir()) == {marker}


def test_previous_configuration_symlink_is_rejected_without_reading_or_overwriting_target(
    source, simulated_runtime, tmp_path,
):
    output = build_portable.build_portable(source)
    external = tmp_path / "operator-secret.json"
    external.write_text('{"private": "unchanged"}', encoding="utf-8")
    configuration = output / "src/config/slice_part_config.json"
    configuration.unlink()
    configuration.symlink_to(external)

    with pytest.raises(ValueError):
        build_portable.build_portable(source, reset_config=False)

    assert configuration.is_symlink()
    assert external.read_text(encoding="utf-8") == '{"private": "unchanged"}'


def test_source_configuration_symlink_does_not_copy_an_external_directory(
    source, simulated_runtime, tmp_path,
):
    output = build_portable.build_portable(source)
    configuration = source / "src/config"
    external = tmp_path / "external-config"
    configuration.replace(external)
    configuration.symlink_to(external, target_is_directory=True)
    write_json(external / "private.json", {"private": "must not be published"})
    before = artifact_snapshot(source / "dist")

    with pytest.raises(ValueError, match="源配置目录"):
        build_portable.build_portable(source)

    assert artifact_snapshot(source / "dist") == before
    assert not (output / "src/config/private.json").exists()
    assert json.loads((external / "private.json").read_text(encoding="utf-8")) == {
        "private": "must not be published",
    }


def test_validation_failure_leaves_previous_three_artifacts_and_unrelated_dist_files(
    source, simulated_runtime,
):
    build_portable.build_portable(source)
    unrelated = source / "dist/other-project.zip"
    unrelated.write_bytes(b"unrelated package")
    before = artifact_snapshot(source / "dist")
    simulated_runtime[1].side_effect = ValueError("relocated runtime failed validation")

    with pytest.raises(ValueError, match="failed validation"):
        build_portable.build_portable(source)

    assert artifact_snapshot(source / "dist") == before
    assert not list((source / "dist").glob(".autoprintfilm-build-*"))


def test_freeze_failure_preserves_previous_artifacts_and_operator_configuration(source, simulated_runtime):
    output = build_portable.build_portable(source)
    write_json(output / "src/config/logger_config.json", {"base_dir": "EditedOperatorLogs"})
    unrelated = source / "dist/other-project.zip"
    unrelated.write_bytes(b"unrelated package")
    before = artifact_snapshot(source / "dist")
    simulated_runtime[2].side_effect = ValueError("executable freeze failed")

    with pytest.raises(ValueError, match="executable freeze failed"):
        build_portable.build_portable(source)

    assert artifact_snapshot(source / "dist") == before
    assert not list((source / "dist").glob(".autoprintfilm-build-*"))


@pytest.mark.parametrize("failed_artifact", ["archive", "checksum"])
def test_publish_failure_restores_every_previous_artifact(
    source, simulated_runtime, monkeypatch, failed_artifact,
):
    build_portable.build_portable(source)
    unrelated = source / "dist/other-project.zip"
    unrelated.write_bytes(b"unrelated package")
    before = artifact_snapshot(source / "dist")
    (source / "main.py").write_text("new source must not replace the old build", encoding="utf-8")
    name = build_portable.ARCHIVE_NAME + (".sha256" if failed_artifact == "checksum" else "")
    target = source / "dist" / name
    replace = Path.replace

    def fail_selected_publication(path, destination):
        if path.name == name and Path(destination) == target:
            raise OSError("simulated publication failure")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_selected_publication)

    with pytest.raises(OSError, match="simulated publication failure"):
        build_portable.build_portable(source)

    assert artifact_snapshot(source / "dist") == before
    assert not list((source / "dist").glob(".autoprintfilm-build-*"))


def write_tar(path, members):
    with tarfile.open(path, "w:gz") as bundle:
        for name, kind, content in members:
            member = tarfile.TarInfo(name)
            if kind == "file":
                value = content.encode("utf-8")
                member.size = len(value)
                bundle.addfile(member, io.BytesIO(value))
            else:
                member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                member.linkname = content
                bundle.addfile(member)


@pytest.mark.parametrize("unsafe_member", [
    ("/outside/payload", "file", "unexpected"),
    ("python/../outside/payload", "file", "unexpected"),
    ("unexpected-root/payload", "file", "unexpected"),
    ("python/bin/link", "symlink", "../../../../outside"),
    ("python/bin/link", "hardlink", "../../outside"),
])
def test_python_archive_escape_is_rejected_before_any_file_is_extracted(tmp_path, unsafe_member):
    archive = tmp_path / "malicious-python.tar.gz"
    destination = tmp_path / "unpacked"
    write_tar(archive, [("python/safe.txt", "file", "safe"), unsafe_member])

    with pytest.raises(ValueError):
        build_portable.extract_python(archive, destination)

    assert not destination.exists()


def test_combined_archive_links_cannot_escape_the_extraction_directory(tmp_path):
    archive = tmp_path / "malicious-python.tar.gz"
    destination = tmp_path / "unpacked/runtime"
    destination.mkdir(parents=True)
    outside = tmp_path / "unpacked/outside"
    outside.mkdir()
    write_tar(archive, [
        ("python/deep/level/a", "symlink", "../.."),
        ("python/deep/level/a/b", "symlink", "../../outside"),
        ("python/deep/level/a/b/payload", "file", "escaped runtime"),
    ])

    # Both links look internal before extraction. After extracting a, resolving
    # b traverses a back to python/ and its ../.. then escapes the runtime.
    with pytest.raises(tarfile.LinkOutsideDestinationError):
        build_portable.extract_python(archive, destination)

    assert (destination / "python/deep/level/a").is_symlink()
    assert not (outside / "payload").exists()


def test_internal_runtime_links_remain_supported(tmp_path):
    archive = tmp_path / "python.tar.gz"
    destination = tmp_path / "runtime"
    write_tar(archive, [
        ("python/bin/python3.11", "file", "runtime interpreter"),
        ("python/bin/python3", "symlink", "python3.11"),
        ("python/bin/python", "hardlink", "python/bin/python3.11"),
    ])

    build_portable.extract_python(archive, destination)

    assert (destination / "python/bin/python3").is_symlink()
    assert (destination / "python/bin/python3").read_text() == "runtime interpreter"
    assert (destination / "python/bin/python").read_text() == "runtime interpreter"


def test_hardlink_to_relative_symlink_links_the_resolved_file_without_touching_outside_files(tmp_path):
    archive = tmp_path / "python.tar.gz"
    destination = tmp_path / "unpacked/runtime"
    outside = tmp_path / "target"
    outside.write_bytes(b"operator-owned file")
    outside.chmod(0o640)
    os.utime(outside, ns=(1_600_000_000_000_000_000, 1_600_000_001_000_000_000))
    original = outside.stat()
    original_contents = outside.read_bytes()
    write_tar(archive, [
        ("python/target", "file", "bundled runtime file"),
        ("python/a/b/c/link", "symlink", "../../../target"),
        ("python/alias", "hardlink", "python/a/b/c/link"),
    ])

    build_portable.extract_python(archive, destination)

    alias = destination / "python/alias"
    target = destination / "python/target"
    symbolic = destination / "python/a/b/c/link"
    assert alias.is_file() and not alias.is_symlink()
    assert alias.samefile(target)
    assert alias.read_bytes() == b"bundled runtime file"
    assert symbolic.is_symlink()
    assert symbolic.readlink() == Path("../../../target")
    assert symbolic.read_bytes() == b"bundled runtime file"
    # Copying the symlink inode to alias would reinterpret ../../../target from
    # python/ and follow this outside file when tarfile applies mode and mtime.
    assert outside.read_bytes() == original_contents
    after = outside.stat()
    assert after.st_mode == original.st_mode
    assert after.st_mtime_ns == original.st_mtime_ns


@pytest.mark.parametrize("legacy_filter", [False, True])
def test_parent_relative_terminfo_links_remain_valid_on_backported_tar_filters(
    tmp_path, monkeypatch, legacy_filter,
):
    archive = tmp_path / "python.tar.gz"
    destination = tmp_path / "runtime"
    write_tar(archive, [
        ("python/share/terminfo/a/ansi", "file", "terminal definition"),
        ("python/share/terminfo/61/ansi", "symlink", "../a/ansi"),
    ])
    if legacy_filter:
        standard_filter = tarfile.data_filter

        def root_relative_filter(member, directory):
            # Model the affected backport: its symbolic-link check interprets
            # linkname relative to the extraction root instead of its parent.
            if member.issym():
                root = Path(directory).resolve()
                target = (root / member.linkname).resolve()
                if not target.is_relative_to(root):
                    raise tarfile.LinkOutsideDestinationError(member, str(target))
            return standard_filter(member, directory)

        compatibility_filter = Mock(side_effect=root_relative_filter)
        monkeypatch.setattr(tarfile, "data_filter", compatibility_filter)

    build_portable.extract_python(archive, destination)

    link = destination / "python/share/terminfo/61/ansi"
    assert link.is_symlink()
    assert link.readlink() == Path("../a/ansi")
    assert link.read_text(encoding="utf-8") == "terminal definition"
    if legacy_filter:
        assert compatibility_filter.called


def test_runtime_root_cannot_be_a_symlink_to_the_original_cache(tmp_path):
    cache = tmp_path / "original cache/python"
    cache.mkdir(parents=True)
    marker = cache / "keep.txt"
    marker.write_text("cached runtime", encoding="utf-8")
    runtime = tmp_path / "moved bundle/python"
    runtime.parent.mkdir()
    runtime.symlink_to(cache, target_is_directory=True)

    with pytest.raises(ValueError):
        build_portable.validate_runtime_links(runtime)

    assert runtime.is_symlink()
    assert marker.read_text(encoding="utf-8") == "cached runtime"


@pytest.mark.parametrize("kind", ["absolute", "relative"])
def test_runtime_dependencies_cannot_link_outside_the_bundled_runtime(tmp_path, kind):
    runtime = tmp_path / "moved bundle/python"
    (runtime / "lib").mkdir(parents=True)
    external = runtime.parent / "outside-runtime.so"
    external.write_text("external dependency", encoding="utf-8")
    link = runtime / "lib/dependency.so"
    link.symlink_to(external if kind == "absolute" else "../../outside-runtime.so")

    with pytest.raises(ValueError):
        build_portable.validate_runtime_links(runtime)

    assert link.is_symlink()
    assert external.read_text(encoding="utf-8") == "external dependency"


def test_runtime_dependencies_can_use_relative_links_inside_the_runtime(tmp_path):
    runtime = tmp_path / "moved bundle/python"
    (runtime / "bin").mkdir(parents=True)
    interpreter = runtime / "bin/python3.11"
    interpreter.write_text("bundled interpreter", encoding="utf-8")
    (runtime / "bin/python3").symlink_to("python3.11")

    build_portable.validate_runtime_links(runtime)


def test_install_command_is_idempotent_for_its_own_symlink(source, tmp_path):
    destination = tmp_path / "user-bin/dev"

    first = build_portable.install_command(source, destination=destination)
    second = build_portable.install_command(source, destination=destination)

    assert first == second == destination
    assert destination.is_symlink()
    assert destination.resolve() == source / "dev"


@pytest.mark.parametrize("existing", ["file", "directory", "other-symlink", "broken-symlink"])
def test_install_command_never_overwrites_a_conflicting_command(source, tmp_path, existing):
    destination = tmp_path / "user-bin/dev"
    destination.parent.mkdir()
    other = tmp_path / "other-command"
    if existing == "file":
        destination.write_text("operator command", encoding="utf-8")
    elif existing == "directory":
        destination.mkdir()
        (destination / "keep.txt").write_text("operator files", encoding="utf-8")
    else:
        if existing == "other-symlink":
            other.write_text("another command", encoding="utf-8")
        destination.symlink_to(other)

    with pytest.raises(ValueError, match="不会覆盖"):
        build_portable.install_command(source, destination=destination)

    if existing == "file":
        assert destination.read_text(encoding="utf-8") == "operator command"
    elif existing == "directory":
        assert (destination / "keep.txt").read_text(encoding="utf-8") == "operator files"
    else:
        assert destination.is_symlink()
        assert destination.readlink() == other


@pytest.mark.parametrize(
    "changed", ["requirements", "build_requirements", "constraints", "python_archive", "python_version"],
)
def test_dependency_or_runtime_changes_invalidate_the_cache_key(source, changed):
    settings = build_portable.build_settings(source)
    original = build_portable.runtime_key(source, settings)
    if changed in {"requirements", "build_requirements", "constraints"}:
        filename = {
            "requirements": "requirements.txt",
            "build_requirements": "requirements-build.txt",
            "constraints": "constraints-build.txt",
        }[changed]
        with (source / filename).open("a", encoding="utf-8") as stream:
            stream.write("additional-dependency==2.0\n")
    elif changed == "python_archive":
        settings["python_sha256"] = "b" * 64
    else:
        settings["python_version"] = "3.11.18"

    assert build_portable.runtime_key(source, settings) != original


def test_cache_key_is_stable_for_settings_order_and_application_only_changes(source):
    settings = build_portable.build_settings(source)
    original = build_portable.runtime_key(source, settings)
    reordered = dict(reversed(list(settings.items())))
    (source / "main.py").write_text("changed application implementation", encoding="utf-8")
    write_json(source / "src/config/slice_part_config.json", {"operator": "new configuration"})

    assert build_portable.runtime_key(source, reordered) == original
