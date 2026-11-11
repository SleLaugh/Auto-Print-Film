"""Resolve an external nnUNet process without importing its inference dependencies."""

import ast
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
from typing import Callable, Optional

from .errors import DependencyError
from src.runtime import application_root


DEFAULT_PREDICT_COMMAND = "nnUNetv2_predict"
GPU_PREDICT_COMMAND = "src/inference/nnunet_gpu_export.py"
EXTERNAL_PREDICT_CODE = (
    "import sys; "
    "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; "
    "sys.argv[0] = 'nnUNetv2_predict'; sys.exit(predict_entry_point())"
)
EXPORT_THREADS_ENV = "AUTOPRINTFILM_NNUNET_EXPORT_THREADS"
RESAMPLING_DEVICE_ENV = "AUTOPRINTFILM_NNUNET_RESAMPLING_DEVICE"


def _adapter_path() -> Path:
    """The external interpreter needs a real script, including frozen releases."""
    if getattr(sys, "frozen", False):
        bundle = Path(getattr(sys, "_MEIPASS", application_root() / "_internal"))
        return bundle / "nnunet_adapter" / "nnunet_export.py"
    return Path(__file__).resolve().parents[1] / "inference" / "nnunet_export.py"


def gpu_adapter_path() -> Path:
    """Locate the bundled GPU entry script in source and executable releases."""
    return _adapter_path().with_name("nnunet_gpu_export.py")


def gpu_adapter_sources() -> tuple[Path, ...]:
    """Include every local module imported by the external GPU entry point."""
    directory = gpu_adapter_path().parent
    return tuple(directory / name for name in (
        "nnunet_gpu_export.py", "nnunet_gpu_resampling.py", "nnunet_export.py",
    ))


def _same_syntax(node: ast.AST, source: str) -> bool:
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(source).body[0], include_attributes=False,
    )


def _official_launcher_source(location: str) -> Optional[str]:
    """Recognize simple pip launchers without dropping custom wrapper behavior.

    Replacing a script merely containing the official import could silently skip
    its environment setup or other work. Accept only the standard import/main
    boilerplate; anything else keeps the configured command unchanged.
    """
    try:
        with Path(location).open("r", encoding="utf-8-sig") as stream:
            source = stream.read(65537)
        if len(source) > 65536:
            return None
        statements = ast.parse(source).body
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return None
    import_seen = False
    main_seen = False
    for node in statements:
        if isinstance(node, ast.Import) and all(
            alias.name in {"re", "sys"} and alias.asname is None for alias in node.names
        ):
            continue
        if _same_syntax(node, "from nnunetv2.inference.predict_from_raw_data import predict_entry_point"):
            import_seen = True
            continue
        if not isinstance(node, ast.If) or main_seen or node.orelse:
            return None
        if ast.dump(node.test) != ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body):
            return None
        body = node.body
        if not body or not _same_syntax(body[-1], "sys.exit(predict_entry_point())"):
            return None
        for statement in body[:-1]:
            if not any(_same_syntax(statement, candidate) for candidate in (
                "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])",
                "sys.argv[0] = sys.argv[0].removesuffix('.exe')",
            )):
                return None
        main_seen = True
    return source if import_seen and main_seen else None


def _launcher_python(source: str) -> Optional[list[str]]:
    """Resolve the Python selected by a verified launcher's shebang.

    Keep venv symlinks and interpreter flags. Unknown env options/assignments or
    non-Python launchers fall back instead of guessing the external environment.
    """
    first_line = source.splitlines()[0] if source else ""
    if not first_line.startswith("#!"):
        return None
    try:
        words = shlex.split(first_line[2:])
    except ValueError:
        return None
    if not words:
        return None
    uses_env = Path(words[0]).name == "env"
    if not Path(words[0]).is_absolute():
        return None
    if uses_env:
        words = words[1:]
        if words and words[0] == "-S":
            words = words[1:]
        elif len(words) != 1:
            return None
    if not words or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(words[0]).name):
        return None
    if not uses_env and not Path(words[0]).is_absolute():
        return None
    # Preserve recognized startup flags; -c/-m/-x and unknown options can change
    # the script being executed or skip its first source line.
    if any(flag not in {"-u", "-O", "-OO", "-B", "-E", "-I", "-s", "-S"} for flag in words[1:]):
        return None
    interpreter = shutil.which(words[0])
    if interpreter is None:
        return None
    return [interpreter, *words[1:]]


def nnunet_inference_launch(
    predict_command: str = DEFAULT_PREDICT_COMMAND,
    python_executable: str = "",
    export_resampling_threads: int = 1,
) -> tuple[list[str], bool, str]:
    """Return a command prefix, adapter status and an observable fallback reason."""
    original = nnunet_command_prefix(predict_command, python_executable)
    if python_executable and Path(original[-1]) == gpu_adapter_path():
        # The GPU entry already installs the export adapter. Keep its command
        # while passing the configured CPU channel count to fallback workers.
        return original, True, ""
    if export_resampling_threads == 1:
        return original, False, ""
    interpreter = None
    if predict_command == DEFAULT_PREDICT_COMMAND and python_executable:
        # This existing configuration already selects the official entry point.
        interpreter = [_location(python_executable)]
    else:
        command = _location(predict_command)
        location = command if os.path.dirname(command) else shutil.which(command)
        source = _official_launcher_source(location) if location else None
        if source is not None:
            interpreter = (
                [_location(python_executable)] if python_executable else _launcher_python(source)
            )
    if interpreter is None:
        return original, False, (
            "无法确认官方 Python 入口或其推理解释器；保留原命令。"
            "如需启用，请配置 nnUNet_python 并使用默认 nnUNetv2_predict，"
            "或配置标准官方 Python launcher"
        )
    adapter = _adapter_path()
    if not adapter.is_file():
        return original, False, f"未找到导出适配脚本 {adapter}；请检查部署包"
    return [*interpreter, str(adapter)], True, ""


def _location(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise DependencyError("nnUNet 外部命令或解释器必须是非空字符串")
    expanded = os.path.expanduser(os.path.expandvars(value))
    if getattr(sys, "frozen", False) and expanded == GPU_PREDICT_COMMAND:
        return str(gpu_adapter_path())
    if os.path.dirname(expanded):
        path = Path(expanded)
        if not path.is_absolute():
            path = application_root() / path
        # Keep interpreter symlinks intact: resolving a venv Python bypasses it.
        return os.path.abspath(path)
    return expanded


def nnunet_command_prefix(
    predict_command: str = DEFAULT_PREDICT_COMMAND,
    python_executable: str = "",
) -> list[str]:
    """Use the external CLI, or an explicitly selected external Python environment.

    With only a Python configured, invoke nnUNet's official entry function there.
    With both fields configured, the command must be a Python script; shell
    wrappers should be executed directly without ``nnUNet_python``.
    """
    command = _location(predict_command)
    if not python_executable:
        return [command]
    interpreter = _location(python_executable)
    if command == DEFAULT_PREDICT_COMMAND:
        return [interpreter, "-c", EXTERNAL_PREDICT_CODE]
    if not os.path.dirname(command):
        command = shutil.which(command) or command
    _reject_shell_script(command)
    return [interpreter, command]


def _reject_shell_script(location: str) -> None:
    path = Path(location)
    if not path.is_file():
        return
    with path.open("rb") as stream:
        first_line = stream.readline(4096).decode("utf-8", errors="replace").strip()
    if not first_line.startswith("#!"):
        return
    words = shlex.split(first_line[2:].strip())
    shells = {"sh", "bash", "dash", "zsh", "ksh", "fish", "csh", "tcsh"}
    if words and (
        Path(words[0]).name in shells
        or (Path(words[0]).name == "env" and any(Path(word).name in shells for word in words[1:]))
    ):
        raise DependencyError(
            "nnUNet_predict_command 是 shell 包装脚本，请清空 nnUNet_python 后直接执行该命令"
        )


def _validate_location(
    value: str,
    *,
    executable: bool,
    executable_finder: Callable[[str], Optional[str]],
) -> str:
    location = _location(value)
    if os.path.dirname(location):
        path = Path(location)
        access = os.X_OK if executable else os.R_OK
        if not path.is_file() or not os.access(path, access):
            kind = "可执行文件" if executable else "Python 脚本"
            raise DependencyError(f"nnUNet 外部{kind}不存在或不可访问: {location}")
        return location
    found = executable_finder(location)
    if found is None:
        raise DependencyError(f"未找到 {location} 可执行命令")
    return found


def validate_nnunet_command(
    executable_finder: Callable[[str], Optional[str]] = shutil.which,
    *,
    predict_command: str = DEFAULT_PREDICT_COMMAND,
    python_executable: str = "",
) -> None:
    """Check external paths only; nnUNet/torch belong to the inference environment."""
    if python_executable:
        _validate_location(
            python_executable, executable=True, executable_finder=executable_finder,
        )
        if predict_command != DEFAULT_PREDICT_COMMAND:
            script = _validate_location(
                predict_command, executable=False, executable_finder=executable_finder,
            )
            _reject_shell_script(script)
        return
    _validate_location(predict_command, executable=True, executable_finder=executable_finder)
