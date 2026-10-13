#!/usr/bin/env python3
"""Freeze a Linux executable with editable JSON and external nnUNet inference."""

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION = "AutoPrintFilm"
ARCHIVE_NAME = "AutoPrintFilm-linux-x64.tar.gz"
BUILD_TOOL = "AutoPrintFilm dev build"
CONFIG_DIRECTORY = Path("src/config")


def _log(message):
    print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(source_root: Path) -> dict:
    """Record source provenance without copying Git output into the release."""
    try:
        repository_root = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not repository_root or Path(repository_root).resolve() != source_root.resolve():
            return {"head": None, "dirty": False, "available": False}
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": False, "available": False}
    return {"head": head, "dirty": bool(status.strip()), "available": True}


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"必须是 JSON 对象：{path}")
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_directory(path):
    if path.is_symlink() or path.absolute() != path.resolve():
        raise ValueError(f"构建目录不能通过符号链接引用：{path}")
    path.mkdir(parents=True, exist_ok=True)


def build_settings(source):
    settings = _json(source / "build-config.json")
    for name in ("python_version", "python_url", "python_sha256"):
        if not isinstance(settings.get(name), str) or not settings[name]:
            raise ValueError(f"build-config.json 缺少 {name}")
    digest = settings["python_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Python 归档 SHA256 格式无效")
    if not settings["python_url"].startswith("https://"):
        raise ValueError("Python 下载地址必须使用 HTTPS")
    (source / "constraints-build.txt").read_text(encoding="utf-8")
    return settings


def runtime_key(source, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for name in ("requirements.txt", "constraints-build.txt", "requirements-build.txt"):
        digest.update(name.encode("ascii"))
        digest.update((source / name).read_bytes())
    return digest.hexdigest()[:24]


def runtime_environment(runtime):
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LD_PRELOAD",
                 "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        environment.pop(name, None)
    environment.update(
        PATH=f"{runtime / 'bin'}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        PYTHONUTF8="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def _run(command, runtime, *, cwd=None, capture=False):
    return subprocess.run(
        [str(value) for value in command], cwd=cwd, env=runtime_environment(runtime),
        check=True, text=True, encoding="utf-8", capture_output=capture,
    )


def download_python(cache, settings):
    downloads = cache / "downloads"
    _safe_directory(downloads)
    archive = downloads / f"python-{settings['python_sha256']}.tar.gz"
    if archive.is_symlink():
        raise ValueError("Python 下载缓存不能是符号链接")
    if archive.is_file() and _sha256(archive) == settings["python_sha256"]:
        return archive
    _log(f"下载 Python {settings['python_version']}（首次构建）…")
    with tempfile.NamedTemporaryFile(dir=downloads, prefix=".download-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            request = urllib.request.Request(settings["python_url"], headers={"User-Agent": BUILD_TOOL})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, stream)
            stream.close()
            if _sha256(temporary) != settings["python_sha256"]:
                raise ValueError("Python 归档 SHA256 校验失败")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    return archive


def _filter_runtime_member(member, destination):
    """Validate symbolic targets relative to their directory on all Python versions.

    Some Ubuntu Python 3.10 data_filter backports interpret symbolic link targets
    as archive-root relative. Validate a normalized hard-link copy, then restore
    the original symbolic link; actual extraction still rechecks each member.
    """
    if member.issym() or member.islnk():
        root = Path(destination).resolve()
        parent = root / PurePosixPath(member.name).parent if member.issym() else root
        linked = (parent / member.linkname).resolve()
        if Path(member.linkname).is_absolute() or not linked.is_relative_to(root / "python"):
            raise tarfile.LinkOutsideDestinationError(member, str(linked))
        normalized = copy.copy(member)
        normalized.linkname = linked.relative_to(root).as_posix()
        if member.issym():
            normalized.type = tarfile.LNKTYPE
            filtered = tarfile.data_filter(normalized, destination)
            filtered.type = tarfile.SYMTYPE
            filtered.linkname = member.linkname
            filtered.mode = None
            return filtered
        return tarfile.data_filter(normalized, destination)
    return tarfile.data_filter(member, destination)


def extract_python(archive, destination):
    """Only accept the expected runtime tree and links that remain inside it."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        root = destination.resolve()
        runtime_root = root / "python"
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != "python":
                raise ValueError(f"Python 归档包含非法路径：{member.name}")
            target = destination / member.name
            if not target.resolve().is_relative_to(root):
                raise ValueError("Python 归档路径越界")
            if member.issym() or member.islnk():
                linked = (target.parent if member.issym() else destination) / member.linkname
                if Path(member.linkname).is_absolute() or not linked.resolve().is_relative_to(runtime_root):
                    raise ValueError("Python 归档链接越界")
            elif not (member.isfile() or member.isdir()):
                raise ValueError("Python 归档包含不支持的文件类型")
        if not hasattr(tarfile, "data_filter"):
            raise ValueError("构建 Python 缺少安全归档过滤器，请使用已更新的 Python 3.10/3.11")
        bundle.extractall(destination, members=members, filter=_filter_runtime_member)


def validate_runtime_links(runtime):
    if runtime.is_symlink() or not runtime.is_dir() or runtime.absolute() != runtime.resolve():
        raise ValueError("运行时根目录必须是包内普通目录")
    root = runtime.resolve()
    for directory, directories, files in os.walk(runtime):
        for name in directories + files:
            path = Path(directory) / name
            if path.is_symlink():
                if Path(os.readlink(path)).is_absolute() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"运行时链接不能指向包外：{path}")


def ensure_runtime(source, settings, cache, *, refresh=False):
    key = runtime_key(source, settings)
    runtime_cache = cache / "runtime"
    _safe_directory(runtime_cache)
    installed = runtime_cache / key
    ready = installed / "READY.json"
    if installed.is_symlink():
        raise ValueError("运行时缓存不能是符号链接")
    if not refresh and ready.is_file() and _json(ready).get("key") == key:
        validate_runtime_links(installed / "python")
        _log("复用独立构建环境缓存。")
        return installed / "python"
    archive = download_python(cache, settings)
    temporary = Path(tempfile.mkdtemp(prefix=".install-", dir=runtime_cache))
    try:
        extract_python(archive, temporary)
        runtime = temporary / "python"
        python = runtime / "bin/python3"
        if not python.is_file():
            raise ValueError("Python 归档缺少 bin/python3")
        _log("准备应用依赖和打包工具（首次构建可能需要较长时间）…")
        pip = [python, "-m", "pip", "--isolated", "install", "--cache-dir", cache / "pip"]
        _run(pip + ["-r", source / "requirements.txt", "-r", source / "requirements-build.txt",
                    "-c", source / "constraints-build.txt"], runtime)
        _run([python, "-m", "pip", "--isolated", "check"], runtime)
        validate_runtime_links(runtime)
        _write_json(temporary / "READY.json", {"key": key, "settings": settings})
        if installed.exists():
            if not ready.is_file():
                raise ValueError("未知运行时目录，不覆盖；请使用新的缓存目录")
            shutil.rmtree(installed)
        temporary.replace(installed)
        return installed / "python"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_existing_output(output):
    if not output.exists() and not output.is_symlink():
        return
    if output.is_symlink() or not output.is_dir():
        raise ValueError("dist/AutoPrintFilm 不是可替换的构建目录")
    marker = output / "PORTABLE.json"
    if marker.is_symlink() or not marker.is_file() or _json(marker).get("build_tool") != BUILD_TOOL:
        raise ValueError("dist/AutoPrintFilm 不是本工具生成的目录，不会覆盖")


def copy_configs(source, application, previous=None):
    """Keep editable JSON, including operator changes in an earlier dist bundle."""
    directory = source / CONFIG_DIRECTORY
    if not directory.is_dir() or directory.absolute() != directory.resolve():
        raise ValueError("源配置目录必须是项目内普通目录")
    (application / CONFIG_DIRECTORY).mkdir(parents=True, exist_ok=True)
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("源配置必须是普通文件")
        shutil.copy2(path, application / CONFIG_DIRECTORY / path.name)
    if previous is not None and previous.exists():
        directory = previous / CONFIG_DIRECTORY
        if directory.is_symlink():
            raise ValueError("已有发布配置目录不能是符号链接")
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
                raise ValueError("已有发布配置必须是普通文件")
            # Malformed edited JSON should fail, rather than silently disappear.
            json.loads(path.read_text(encoding="utf-8"))
            shutil.copy2(path, application / CONFIG_DIRECTORY / path.name)


def write_manifest(application, source, settings, key):
    files = [path.relative_to(application).as_posix() for path in application.rglob("*")
             if path.is_file() and path.name not in {"RELEASE.json", "PORTABLE.json"}]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(source),
        "packaging": "pyinstaller-onedir",
        "executable": APPLICATION,
    }
    manifest["files"] = {
        relative: _sha256(application / relative)
        for relative in sorted(files)
    }
    _write_json(application / "RELEASE.json", manifest)
    _write_json(application / "PORTABLE.json", {
        "schema_version": 2, "build_tool": BUILD_TOOL, "runtime_dir": "_internal",
        "packaging": "pyinstaller-onedir", "executable": APPLICATION,
        "platform": "linux-x86_64",
        "build_host_glibc": platform.libc_ver()[1],
        "inference_environment": "external",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_key": key, "settings": settings, "editable_config_dir": "src/config",
    })


def freeze_application(source, runtime, temporary):
    """Compile Python modules into an executable bundle, without copying sources."""
    specification = source / "AutoPrintFilm.spec"
    entry = source / "scripts/frozen_entry.py"
    for path in (specification, entry):
        if path.is_symlink() or not path.is_file() or path.absolute() != path.resolve():
            raise ValueError(f"缺少打包所需的普通文件：{path}")
    cache = source / ".build-cache"
    work = cache / "pyinstaller-work"
    configuration = cache / "pyinstaller-config"
    _safe_directory(work)
    _safe_directory(configuration)
    environment = runtime_environment(runtime)
    environment["PYINSTALLER_CONFIG_DIR"] = str(configuration)
    _log("打包应用代码和依赖为可执行程序…")
    result = subprocess.run(
        [str(runtime / "bin/python3"), "-m", "PyInstaller", "--noconfirm",
         "--distpath", str(temporary / "pyinstaller-output"),
         "--workpath", str(work), str(specification)],
        cwd=source, env=environment, text=True, encoding="utf-8", capture_output=True,
    )
    (cache / "last-build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    application = temporary / "pyinstaller-output" / APPLICATION
    if not (application / APPLICATION).is_file():
        raise ValueError("打包工具未生成 AutoPrintFilm 可执行程序")
    frozen = _run([runtime / "bin/python3", "-m", "pip", "--isolated", "freeze"],
                  runtime, capture=True)
    (application / "requirements.lock.txt").write_text(frozen.stdout, encoding="utf-8")
    return application


def validate_application(application):
    validate_runtime_links(application)
    executable = application / APPLICATION
    if not os.access(executable, os.X_OK):
        raise ValueError("发布入口不可执行")
    if (application / "main.py").exists() or (application / "python").exists():
        raise ValueError("发布包不应包含旧源码入口或独立 Python 目录")
    if any((application / "src").rglob("*.py")):
        raise ValueError("应用源码应打入程序，不应复制到发布目录")
    # Launch from outside the executable directory to detect cwd-based config bugs.
    _run([executable, "--help"], application, cwd=application.parent, capture=True)
    checked = _run([executable, "--_build-check"], application,
                   cwd=application.parent, capture=True)
    details = json.loads(checked.stdout.strip().splitlines()[-1])
    if not details.get("frozen") or Path(details["application_root"]).resolve() != application.resolve():
        raise ValueError("程序运行位置与发布目录不一致")
    if Path(details["config_dir"]).resolve() != (application / CONFIG_DIRECTORY).resolve():
        raise ValueError("程序没有读取外置配置")
    logger = application / CONFIG_DIRECTORY / "logger_config.json"
    original = logger.read_bytes()
    changed = _json(logger)
    changed["base_dir"] = "Build check 外置配置"
    try:
        _write_json(logger, changed)
        edited = _run([executable, "--_build-check"], application,
                      cwd=application.parent, capture=True)
        if json.loads(edited.stdout.strip().splitlines()[-1])["logging_base_dir"] != changed["base_dir"]:
            raise ValueError("修改外置 JSON 未生效")
    finally:
        logger.write_bytes(original)


def _publish(application, archive, checksum, dist):
    """Install verified build products, retaining old products on a publish error."""
    targets = ((application, dist / APPLICATION), (archive, dist / ARCHIVE_NAME),
               (checksum, dist / f"{ARCHIVE_NAME}.sha256"))
    backups = []
    published = []
    _validate_existing_output(dist / APPLICATION)
    for _, target in targets:
        if target.is_symlink():
            raise ValueError(f"发布目标不能是符号链接：{target}")
        if target != dist / APPLICATION and target.exists() and not target.is_file():
            raise ValueError(f"压缩包目标必须是普通文件：{target}")
    try:
        for incoming, target in targets:
            if target.is_symlink():
                raise ValueError(f"发布目标不能是符号链接：{target}")
            if target.exists():
                backup = incoming.parent / f".previous-{target.name}"
                target.replace(backup)
                backups.append((backup, target))
            incoming.replace(target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        for backup, target in reversed(backups):
            backup.replace(target)
        raise


@contextmanager
def build_lock(cache):
    path = cache / "build.lock"
    if path.is_symlink():
        raise ValueError("构建锁不能是符号链接")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("已有 dev build 正在运行，请等待它结束") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def build_portable(source=PROJECT_ROOT, *, refresh=False, reset_config=False):
    source = Path(source).resolve()
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("当前发布配置支持 Linux x86_64，请在相同平台构建")
    settings = build_settings(source)
    cache = source / ".build-cache"
    dist = source / "dist"
    _safe_directory(cache)
    _safe_directory(dist)
    output = dist / APPLICATION
    _validate_existing_output(output)
    with build_lock(cache):
        runtime = ensure_runtime(source, settings, cache, refresh=refresh)
        temporary = Path(tempfile.mkdtemp(prefix=".autoprintfilm-build-", dir=dist))
        try:
            _log("构建直接运行的应用程序与外置配置…")
            # Deliberately use a different path for validation before the final
            # dist move, so absolute build-time runtime paths cannot pass unnoticed.
            staging = freeze_application(source, runtime, temporary)
            copy_configs(source, staging, None if reset_config else output)
            for relative in ("README.md", "docs/production.md"):
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
            application = temporary / "relocated bundle" / APPLICATION
            application.parent.mkdir()
            staging.replace(application)
            _log("验证移动后直接运行、影像依赖和外置配置（不运行推理）…")
            validate_application(application)
            write_manifest(application, source, settings, runtime_key(source, settings))
            _log("生成压缩包…")
            archive = temporary / ARCHIVE_NAME
            with tarfile.open(archive, "w:gz", compresslevel=3) as bundle:
                bundle.add(application, arcname=APPLICATION)
            checksum = temporary / f"{ARCHIVE_NAME}.sha256"
            checksum.write_text(f"{_sha256(archive)}  {ARCHIVE_NAME}\n", encoding="ascii")
            _publish(application, archive, checksum, dist)
            _log(f"构建完成：{output}")
            _log(f"发布包：{dist / ARCHIVE_NAME}（{(dist / ARCHIVE_NAME).stat().st_size / 1024**2:.1f} MiB）")
            return output
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def install_command(source=PROJECT_ROOT, *, destination=None):
    source = Path(source).resolve()
    target = Path(destination) if destination is not None else Path.home() / ".local/bin/dev"
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source / "dev":
            _log(f"dev 命令已注册：{target}")
            return target
        raise ValueError(f"已有其他 dev 命令，不会覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source / "dev")
    _log(f"dev 命令已注册：{target}")
    if str(target.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        _log('当前 PATH 尚未包含 ~/.local/bin；新开终端或执行 export PATH="$HOME/.local/bin:$PATH"。')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="项目命令：dev build 生成可移动发布包到 dist/")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="构建可执行程序、内部依赖和外置配置")
    build.add_argument("--refresh-runtime", action="store_true", help="重新安装依赖缓存")
    build.add_argument("--reset-config", action="store_true", help="用源码配置替换上次 dist 中的配置")
    commands.add_parser("install-command", help="一次性注册裸命令 dev 到 ~/.local/bin")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "install-command":
            install_command()
        else:
            build_portable(refresh=arguments.refresh_runtime, reset_config=arguments.reset_config)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        _log(f"操作失败：{error}")
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            _log(error.stderr[-6000:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
