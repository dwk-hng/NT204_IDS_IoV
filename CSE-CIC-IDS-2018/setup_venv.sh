#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"

LOG_DIR="$PROJECT_DIR/logs"
CACHE_DIR="$PROJECT_DIR/cache"
TMP_CACHE_DIR="$CACHE_DIR/tmp"
PIP_CACHE_DIR="$CACHE_DIR/pip"
TORCH_CACHE_DIR="$CACHE_DIR/torch"
MPL_CACHE_DIR="$CACHE_DIR/matplotlib"
JOBLIB_CACHE_DIR="$CACHE_DIR/joblib"
XDG_CACHE_DIR="$CACHE_DIR/xdg"

mkdir -p "$LOG_DIR" "$TMP_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_CACHE_DIR" "$MPL_CACHE_DIR" "$JOBLIB_CACHE_DIR" "$XDG_CACHE_DIR"

if command -v module >/dev/null 2>&1; then
    module clear -f || true
    module load shared python312 || true
fi

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ Không tìm thấy python3.12/python3 trong PATH"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"

aif="false"
if [ -d "$VENV_DIR" ]; then
    echo "ℹ️ Venv đã tồn tại tại $VENV_DIR"
    aif="true"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export TMPDIR="$TMP_CACHE_DIR"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TORCH_HOME="$TORCH_CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"
export JOBLIB_TEMP_FOLDER="$JOBLIB_CACHE_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

if [ "$aif" = "true" ]; then
    echo "✅ Đã cập nhật packages cho venv sẵn có."
else
    echo "✅ Đã tạo venv mới và cài packages."
fi

echo "Venv: $VENV_DIR"
echo "Logs: $LOG_DIR"
echo "Cache: $CACHE_DIR"
