"""Locate editable application files and launch external processes safely."""

import os
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping


NATIVE_THREAD_VARIABLES = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_thread_count(value: Any, setting: str) -> int:
    """Reject invalid limits rather than silently changing the requested value."""
    if isinstance(value, bool):
        raise ValueError(f"{setting} 必须是正整数")
    if isinstance(value, int):
        count = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        count = int(value)
    else:
        raise ValueError(f"{setting} 必须是正整数")
    if count < 1:
        raise ValueError(f"{setting} 必须是正整数")
    return count


def _available_cpu_count() -> int:
    """Respect process CPU affinity when available, including restricted workers."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def bootstrap_native_threads(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set CLI defaults before importing NumPy/SimpleITK; preserve explicit limits.

    Calling code must run this bootstrap before importing application stages.
    ITK's default limits each filter operation, not the total process thread
    count. Child processes inherit these environment defaults unless overridden.
    Importing this module alone never changes the process environment. Invalid
    application overrides raise ValueError before changing the environment.
    """
    environment = os.environ if environ is None else environ
    native_override = environment.get("AUTOPRINTFILM_NATIVE_THREADS")
    itk_default = (
        _positive_thread_count(native_override, "AUTOPRINTFILM_NATIVE_THREADS")
        if native_override is not None
        else min(4, _available_cpu_count())
    )
    defaults = {name: "1" for name in NATIVE_THREAD_VARIABLES}
    defaults["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_default)
    # Native libraries support their own syntax (for example OMP's "4,2").
    # Preserve explicit values verbatim and validate only our application knob.
    effective = {
        name: environment.get(name, default)
        for name, default in defaults.items()
    }
    for name, value in effective.items():
        environment.setdefault(name, value)
    return effective


def window_worker_count(
    config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Resolve constructor override, environment override, then two workers.

    This limits concurrent window tasks, not native threads used by each task.
    """
    if config is not None and "max_window_workers" in config:
        return _positive_thread_count(config["max_window_workers"], "max_window_workers")
    environment = os.environ if environ is None else environ
    return _positive_thread_count(
        environment.get("AUTOPRINTFILM_WINDOW_WORKERS", "2"),
        "AUTOPRINTFILM_WINDOW_WORKERS",
    )


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
