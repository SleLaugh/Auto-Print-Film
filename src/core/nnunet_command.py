"""Resolve an external nnUNet process without importing its inference dependencies."""

import os
from pathlib import Path
import shlex
import shutil
from typing import Callable, Optional

from .errors import DependencyError
from src.runtime import application_root


DEFAULT_PREDICT_COMMAND = "nnUNetv2_predict"
EXTERNAL_PREDICT_CODE = (
    "import sys; "
    "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; "
    "sys.argv[0] = 'nnUNetv2_predict'; sys.exit(predict_entry_point())"
)


def _location(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise DependencyError("nnUNet 外部命令或解释器必须是非空字符串")
    expanded = os.path.expanduser(os.path.expandvars(value))
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
