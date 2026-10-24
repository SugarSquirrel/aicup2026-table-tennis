#!/usr/bin/env bash
# Re-run the model pipeline for the paper's post-competition experiments.
# Usage (repo root): bash paper_analysis/run_experiments.sh [step ...]
#   steps: v7 v8 v14 v14g oof   (default: all, in this order; each later step needs the v7/v8 OOF caches)
# Needs the uv env (.venv), data/ and a TabPFN token in $TABPFN_TOKEN or tabpfn_apikey.txt (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
LOG="$ROOT/paper_analysis/logs"; mkdir -p "$LOG"
if [ -z "${TABPFN_TOKEN:-}" ] && [ -f "$ROOT/tabpfn_apikey.txt" ]; then
  export TABPFN_TOKEN="$(tr -d '[:space:]' < "$ROOT/tabpfn_apikey.txt")"
fi
[ -n "${TABPFN_TOKEN:-}" ] || { echo "TABPFN_TOKEN not set and tabpfn_apikey.txt missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

declare -A SCRIPT=(
  [v7]=gen_submission_v7_actionaug.py
  [v8]=gen_submission_v8_t2v_film.py
  [v14]=gen_submission_v14_old_in_training.py
  [v14g]=gen_submission_v14g_gated_old_training.py
  [oof]=oof_ablation_v10_v14.py
)
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(v7 v8 v14 v14g oof)

cd "$ROOT/src"   # scripts use ../data and write ../submission_*.csv
for s in "${STEPS[@]}"; do
  t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] START $s (${SCRIPT[$s]})"
  if "$PY" "${SCRIPT[$s]}" > "$LOG/$s.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE  $s in $(( $(date +%s) - t0 ))s"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $s (see paper_analysis/logs/$s.log)"; tail -20 "$LOG/$s.log"; exit 1
  fi
done
echo "ALL DONE"
