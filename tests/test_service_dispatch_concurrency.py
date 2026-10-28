"""Compile actual service admission methods and exercise harmless real children.

The C# service project and its dependencies live outside this repository. This
harness substitutes the database/configuration and startup recovery only, keeping
the actual admission, process launch, tracking and task-status methods intact.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dotnet_sdk():
    for candidate in dict.fromkeys((shutil.which("dotnet"), "/usr/bin/dotnet")):
        if not candidate or not Path(candidate).is_file():
            continue
        result = subprocess.run(
            [candidate, "--list-sdks"], capture_output=True, text=True, timeout=10
        )
        versions = re.findall(r"^(\d+)\.\d+\.\d+", result.stdout, re.MULTILINE)
        if result.returncode == 0 and versions and int(versions[-1]) >= 6:
            return candidate, int(versions[-1])
    pytest.skip("A .NET SDK >= 6 is required for the C# admission regression test")


def _method(source, name):
    """Extract an entire actual C# method without rewriting its implementation."""
    match = re.search(rf"^        private [^\n]*\b{name}\(", source, re.MULTILINE)
    assert match is not None, f"Service method {name} is missing"
    body_start = source.index("{", match.end())
    depth = 1
    end = body_start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


@pytest.mark.skipif(os.name == "nt", reason="The harmless worker uses a POSIX shell")
def test_parallel_service_dispatch_obeys_output_and_global_limits(tmp_path):
    dotnet, sdk_major = _dotnet_sdk()
    source = (PROJECT_ROOT / "服务端/AISliceDicomService.cs").read_text("utf-8")
    methods = "\n\n".join(
        _method(source, name)
        for name in (
            "ProcessSingleStudy",
            "StartUnprocessedTasks",
            "StartUnprocessedSeries",
            "UpdateSeriesRecordStatus",
            "IsProcessRunning",
            "TryGetRunningProcessStartTime",
            "GetSeriesCacheDirectory",
            "HasReusableSeriesCache",
            "NormalizeOutputDirectory",
            "IsOutputProcessRunning",
            "TrackRunningProcess",
            "RefreshRunningProcesses",
            "StartAutoPrintFilm",
        )
    )
    harness = (PROJECT_ROOT / "tests/fixtures/service_dispatch_harness.cs").read_text(
        "utf-8"
    )
    (tmp_path / "Program.cs").write_text(
        harness.replace("// ACTUAL_SERVICE_METHODS", methods), encoding="utf-8"
    )
    (tmp_path / "Harness.csproj").write_text(
        '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
        '<OutputType>Exe</OutputType>'
        f'<TargetFramework>net{sdk_major}.0</TargetFramework>'
        '<ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable>'
        '</PropertyGroup></Project>',
        encoding="utf-8",
    )
    # No external packages or NuGet connections are needed.
    (tmp_path / "NuGet.Config").write_text(
        '<configuration><packageSources><clear /></packageSources></configuration>',
        encoding="utf-8",
    )
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    worker.chmod(0o700)
    environment = {
        **os.environ,
        "DOTNET_CLI_HOME": str(tmp_path / "dotnet-home"),
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_NOLOGO": "1",
    }
    build = subprocess.run(
        [dotnet, "build", "Harness.csproj", "--nologo", "--verbosity", "quiet"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [dotnet, str(tmp_path / f"bin/Debug/net{sdk_major}.0/Harness.dll"), str(worker)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "All 7 dispatch scenarios passed" in run.stdout
    assert "Series reuse invocation arguments passed" in run.stdout
