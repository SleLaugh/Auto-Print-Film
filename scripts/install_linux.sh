#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

# 创建项目专用虚拟环境，避免污染系统 Python 环境
"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate

# 先升级安装工具，再根据 requirements.txt 安装项目依赖
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "依赖已安装到 .venv"
echo "激活虚拟环境命令: source .venv/bin/activate"
