#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "未找到 .venv/bin/python，请先运行 scripts/install_linux.sh" >&2
    exit 1
fi

.venv/bin/python -m unittest discover -s tests -v
