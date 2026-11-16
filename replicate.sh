#!/usr/bin/env bash
# replicate.sh — reproduce the full audit pipeline in order.
#
# Requirements: uv (or any Python 3.11+), an LLM API key in ADAPTION_API_KEY.
# All results go to timestamped run dirs under results/runs/; reruns never
# overwrite prior runs unless --force is passed to the runners.
#
# Usage:
#   export ADAPTION_API_KEY=...   # or source your shell profile
#   ./replicate.sh                # full pipeline
#   STEP=anchors ./replicate.sh   # single step: setup|smoke|redteam|judgemeta|anchors|evasion

set -euo pipefail
cd "$(dirname "$0")"
STEP="${STEP:-all}"

if [ -z "${ADAPTION_API_KEY:-}" ]; then
  echo "ERROR: ADAPTION_API_KEY not set (never commit or echo it)." >&2
  exit 1
fi

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "=== setup: minimal venv (uv venv; no numpy/scipy needed) ==="
  uv venv .venv
  uv pip install --python .venv/bin/python adaption httpx inspect-ai pydantic
fi

run() { echo "=== $1 ==="; shift; "$PY" -u "$@"; }

[ "$STEP" = "all" ] || [ "$STEP" = "smoke" ] && {
  run smoke experiments/smoke_test.py 2>/dev/null || echo "(smoke test optional; skipping if absent)"
}

[ "$STEP" = "all" ] || [ "$STEP" = "redteam" ] && {
  run "01 red-team: native+hardened, crafted attacks" experiments/01_run_redteam.py --n-tasks 30 --workers 20 --mode both
}

[ "$STEP" = "all" ] || [ "$STEP" = "judgemeta" ] && {
  run "02 judge meta-eval: inter-judge vs reference" experiments/02_run_judge_meta.py --n-per-benchmark 85 --workers 24 --mode both
}

[ "$STEP" = "all" ] || [ "$STEP" = "anchors" ] && {
  run "03 anchors + real-model calibration" experiments/03_run_anchors.py --n-per-benchmark 40 --workers 8
}

[ "$STEP" = "all" ] || [ "$STEP" = "evasion" ] && {
  run "05 evasion system-prompt arm (resumable)" experiments/05_evasion_system_prompt.py --n 40 --workers 8
}

echo "=== done. results under results/runs/<timestamp>_<label>/ ; per-item evidence alongside every aggregate ==="
