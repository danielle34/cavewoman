#!/bin/bash
# Size-ablation sweep job body: one (model size, condition) across
# 5 datasets x levels {L0,L1,L2,L4}, n=500 seeded items each.
#
# Env (via sbatch --export): SWEEP_TAG, SWEEP_PATH, SWEEP_COND
#   optional SWEEP_EXTRA: extra flags appended to run_experiment.py
#   (e.g. "--disable_thinking" for Qwen3-family models).
# run_experiment.py is resume-safe per (dataset, level) JSONL, so
# preemption + --requeue restarts exactly where it stopped.

set -u

# Site configuration: override via environment, or edit here.
#   CAVEWOMAN_ENV_SETUP  shell snippet to activate your Python env (default: none)
#   CAVEWOMAN_ROOT       path to this repo checkout (default: script's repo root)
: "${CAVEWOMAN_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ -n "${CAVEWOMAN_ENV_SETUP:-}" ]; then eval "$CAVEWOMAN_ENV_SETUP"; fi
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

echo "=== size sweep job ==="
date -Iseconds; hostname; nvidia-smi -L 2>&1 | head -4
echo "  tag=$SWEEP_TAG path=$SWEEP_PATH cond=$SWEEP_COND restart=${SLURM_RESTART_COUNT:-0}"

cd "$CAVEWOMAN_ROOT"

N_ITEMS="${SWEEP_N:-500}"
SEED="${SWEEP_SEED:-42}"
DATASETS=(${SWEEP_DATASETS:-gsm8k boolq arc_easy commonsenseqa mmlu_stem})
LEVELS=(${SWEEP_LEVELS:-L0 L1 L2 L4})

# SWEEP_N=full (or empty) omits --n entirely, which run_experiment.py reads as
# "use the whole split".
N_FLAG=(--n "$N_ITEMS")
if [ -z "$N_ITEMS" ] || [ "$N_ITEMS" = "full" ]; then
    N_FLAG=()
fi

RC_TOTAL=0
for DS in "${DATASETS[@]}"; do
    OUT="results/${SWEEP_TAG}_${SWEEP_COND}/${DS}"
    mkdir -p "$OUT"
    for LV in "${LEVELS[@]}"; do
        echo "--- $SWEEP_TAG $SWEEP_COND $DS $LV ($(date -Iseconds)) ---"
        python experiments/run_experiment.py \
            --model_name "$SWEEP_TAG" \
            --model_path "$SWEEP_PATH" \
            --dataset "$DS" \
            --condition "$SWEEP_COND" \
            --level "$LV" \
            "${N_FLAG[@]}" \
            --seed "$SEED" \
            --output "$OUT" \
            ${SWEEP_EXTRA:-}
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "[sweep] FAILED $DS $LV rc=$RC"
            RC_TOTAL=$RC
        fi
    done
done

echo "[sweep] job done rc_total=$RC_TOTAL  $(date -Iseconds)"
exit $RC_TOTAL
