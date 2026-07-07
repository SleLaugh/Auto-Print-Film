$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

# 创建项目专用虚拟环境，优先使用 Windows 的 py 启动器
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv .venv
} else {
    python -m venv .venv
}

# 先升级安装工具，再根据 requirements.txt 安装项目依赖
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "依赖已安装到 .venv"
Write-Host "激活虚拟环境命令: .\.venv\Scripts\Activate.ps1"
