#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-rewrite-build}"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt -r requirements-dev.txt
python -m compileall -q split3r_rewrite tests_rewrite
python -m pytest -q tests_rewrite
pyinstaller packaging/split3r_rewrite.spec --clean --noconfirm
printf '\nRewrite build ready at: %s/dist/Split3rRewrite\n' "$ROOT_DIR"
