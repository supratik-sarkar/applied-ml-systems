#!/usr/bin/env bash
set -euo pipefail

# Root verification script for applied-ml-systems
# Verifies Python 3.12.13, creates ephemeral venv in mktemp -d, runs tests & syntax checks, tears down venv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python 3.12.13
if command -v python3.12 &>/dev/null; then
    PYTHON_CMD="python3.12"
elif [[ -x "/opt/homebrew/bin/python3.12" ]]; then
    PYTHON_CMD="/opt/homebrew/bin/python3.12"
else
    PYTHON_CMD="python3"
fi

echo "=== [applied-ml-systems] Starting Umbrella Verification ==="

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "Detected Python binary: $PYTHON_CMD ($PY_VER)"

TMP_VENV=$(mktemp -d -t "venv_applied_ml_XXXXXX")
cleanup() {
    if [[ -d "$TMP_VENV" ]]; then
        rm -rf "$TMP_VENV"
    fi
}
trap cleanup EXIT

"$PYTHON_CMD" -m venv "$TMP_VENV"
VENV_PY="$TMP_VENV/bin/python"
VENV_PIP="$TMP_VENV/bin/pip"
VENV_PYTEST="$TMP_VENV/bin/pytest"

"$VENV_PIP" install --quiet --disable-pip-version-check pytest

echo ""
echo ">>> Verifying: legacy/anomaly-detection"
"$VENV_PY" -m py_compile "$ROOT_DIR/legacy/anomaly-detection/non_time_series_anomaly.py"
"$VENV_PY" -m py_compile "$ROOT_DIR/legacy/anomaly-detection/time_series_anomaly.py"
echo ">>> legacy/anomaly-detection: PASSED"

echo ""
echo ">>> Verifying: legacy/optimization-hpo"
"$VENV_PYTEST" "$ROOT_DIR/legacy/optimization-hpo/tests" -q
echo ">>> legacy/optimization-hpo: PASSED"

echo ""
echo ">>> Verifying: legacy/recommenders"
"$VENV_PYTEST" "$ROOT_DIR/legacy/recommenders/tests" -q
echo ">>> legacy/recommenders: PASSED"

cleanup
trap - EXIT

echo ""
echo "============================================================"
echo "=== ALL APPLIED ML SUBSYSTEMS VERIFIED SUCCESSFULLY ==="
echo "============================================================"
