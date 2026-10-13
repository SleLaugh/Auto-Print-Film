"""Locate editable application files and launch external processes safely."""

import os
from pathlib import Path
import sys
from typing import Mapping


def application_root() -> Path:
    """Keep external configuration beside a frozen executable, outside _internal."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def config_path(filename: str) -> Path:
    """Return the existing src/config layout in source and executable releases."""
    return application_root() / "src" / "config" / filename


def config_candidates(filename: str) -> list[Path]:
    """Preserve development overrides; frozen apps use their editable config only."""
    standard = config_path(filename)
    if getattr(sys, "frozen", False):
        return [standard]
    return [Path.cwd() / "config" / filename, Path.cwd() / filename, standard]


def resolve_application_path(value: str) -> str:
    """展开配置路径；相对路径以程序目录为基准，空值保留环境变量回退。"""
    if not value:
        return ""
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = application_root() / path
    return str(path)


def external_process_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Restore pre-bootloader library paths for external commands on Linux/Unix.

    Pass the returned copy to subprocess; the application's bundled libraries
    must stay available to its own process. Development environments are copied
    unchanged.
    """
    environment = dict(os.environ if environ is None else environ)
    if getattr(sys, "frozen", False):
        if sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd")):
            variable = "LD_LIBRARY_PATH"
        elif sys.platform.startswith("aix"):
            variable = "LIBPATH"
        else:
            return environment
        original = environment.get(variable + "_ORIG")
        if original is None:
            environment.pop(variable, None)
        else:
            environment[variable] = original
    return environment
