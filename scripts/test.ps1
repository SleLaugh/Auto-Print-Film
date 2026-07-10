$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "未找到 $Python，请先运行 scripts/install_windows.ps1"
}

& $Python -m unittest discover -s tests -v
