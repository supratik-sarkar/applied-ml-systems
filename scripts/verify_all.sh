#!/usr/bin/env bash
set -euo pipefail

# Root verification script for applied-ml-systems
# Enforces exact Python 3.12.13 contract, creates ephemeral venv in mktemp -d, runs tests & syntax checks, tears down venv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python executable (allows external PYTHON_CMD override)
PYTHON_CMD="${PYTHON_CMD:-python3.12}"

if ! command -v "$PYTHON_CMD" &>/dev/null; then
    if [[ -x "/opt/homebrew/bin/python3.12" ]]; then
        PYTHON_CMD="/opt/homebrew/bin/python3.12"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if ! command -v "$PYTHON_CMD" &>/dev/null; then
    echo "ERROR: Python executable '$PYTHON_CMD' not found." >&2
    exit 2
fi

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "Detected Python binary: $PYTHON_CMD ($PY_VER)"

# Strict Python contract: must be exactly 3.12.13
if [[ "$PY_VER" != "3.12.13" ]]; then
    echo "ERROR: Python 3.12.13 is required; found $PY_VER." >&2
    exit 2
fi

echo "=== [applied-ml-systems] Starting Umbrella Verification ==="

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
echo ">>> Verifying: legacy/anomaly-detection (2 module syntax checks)"
"$VENV_PY" -m py_compile "$ROOT_DIR/legacy/anomaly-detection/non_time_series_anomaly.py"
"$VENV_PY" -m py_compile "$ROOT_DIR/legacy/anomaly-detection/time_series_anomaly.py"
echo ">>> legacy/anomaly-detection: PASSED (2 modules syntax-compiled)"

echo ""
echo ">>> Verifying: legacy/optimization-hpo"
(cd "$ROOT_DIR/legacy/optimization-hpo" && "$VENV_PYTEST" "tests" -q)
echo ">>> legacy/optimization-hpo: PASSED (1 test passed)"

echo ""
echo ">>> Verifying: legacy/recommenders"
(cd "$ROOT_DIR/legacy/recommenders" && "$VENV_PYTEST" "tests" -q)
echo ">>> legacy/recommenders: PASSED (1 test passed)"

cleanup
trap - EXIT

echo ""
echo "============================================================"
echo "=== APPLIED ML SUBSYSTEMS VERIFIED (2 tests + 2 syntax checks) ==="
echo "============================================================"
