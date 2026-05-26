#!/bin/bash
# Run GPT-5.4 on all 5 datasets × 2 conditions sequentially.
# Designed to be run inside tmux. Skips any (dataset, condition) whose 5 levels
# are all already at the expected record count.

set -u

# Cap OpenBLAS/MKL/OMP thread counts so numpy doesn't fork dozens of
# threads at import. Some shared or multi-user environments enforce a low
# process limit (RLIMIT_NPROC) and the launcher would otherwise die before
# the first API call. Safe for API-bound work that does no heavy numerics.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
MODEL_TAG="gpt-5.4"
MODEL_ID="gpt-5.4-2026-03-05"

mkdir -p "$LOGS_DIR"

# ---- env check ----
# Assumes you have activated a Python environment with the dependencies
# from requirements.txt installed (see ../requirements.txt at the repo root).

# ---- load .env (and verify key) ----
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "[ERROR] $REPO_ROOT/.env not found. Create it with OPENAI_API_KEY=sk-..." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/.env"
set +a
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "your_key_here" ]; then
    echo "[ERROR] OPENAI_API_KEY not set or still placeholder in $REPO_ROOT/.env" >&2
    echo "        Edit the repository .env and replace 'your_key_here' with a real key." >&2
    exit 1
fi
echo "[gpt-5.4] OPENAI_API_KEY loaded (length: ${#OPENAI_API_KEY})"

# ---- expected per-dataset counts (used for completion check) ----
declare -A EXPECTED=(
    [gsm8k]=1319
    [boolq]=3270
    [arc_easy]=2376
    [commonsenseqa]=1221
    [mmlu_stem]=3279
)

DATASETS=(gsm8k boolq arc_easy commonsenseqa mmlu_stem)
# CONDITIONS defaults to "output input"; override via env var:
#   CAVEWOMAN_CONDITIONS=output  bash run_gpt5.4_all.sh   # output-only sweep
#   CAVEWOMAN_CONDITIONS=input   bash run_gpt5.4_all.sh   # input-only sweep
# Useful for parallel runs in two tmux sessions (~2× wall-clock speedup).
read -ra CONDITIONS <<< "${CAVEWOMAN_CONDITIONS:-output input}"
LEVELS=(L0 L1 L2 L3 L4)

echo "===================================================================="
echo "[gpt-5.4] START $(date -Iseconds) on $(hostname)"
echo "  model:      $MODEL_ID (tag: $MODEL_TAG)"
echo "  datasets:   ${DATASETS[*]}"
echo "  conditions: ${CONDITIONS[*]}"
echo "  log dir:    $LOGS_DIR"
echo "===================================================================="

for DATASET in "${DATASETS[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        OUT_DIR="$REPO_ROOT/results/${MODEL_TAG}_${CONDITION}/${DATASET}"
        LOG="$LOGS_DIR/gpt5.4_${DATASET}_${CONDITION}.log"

        echo
        echo "----------------------------------------------------------------"
        echo "[gpt-5.4] $(date -Iseconds)  ${DATASET} / ${CONDITION}"
        echo "----------------------------------------------------------------"

        # Skip-if-complete check across all 5 levels
        ALL_DONE=true
        for LEVEL in "${LEVELS[@]}"; do
            JSONL="$OUT_DIR/caveman_${MODEL_TAG}_${DATASET}_${CONDITION}_${LEVEL}.jsonl"
            if [ -f "$JSONL" ]; then
                COUNT=$(wc -l < "$JSONL" | tr -d ' ')
            else
                COUNT=0
            fi
            if [ "$COUNT" -lt "${EXPECTED[$DATASET]}" ]; then
                ALL_DONE=false
                break
            fi
        done

        if [ "$ALL_DONE" = "true" ]; then
            echo "  [skip] all 5 levels already at ${EXPECTED[$DATASET]} records"
            continue
        fi

        mkdir -p "$OUT_DIR"
        python "$REPO_ROOT/experiments/run_experiment_api.py" \
            --model "$MODEL_ID" \
            --dataset "$DATASET" \
            --condition "$CONDITION" \
            --level all \
            --output_dir "$OUT_DIR" \
            2>&1 | tee -a "$LOG"

        echo "  [done] ${DATASET}/${CONDITION}  log=$LOG"
    done
done

echo
echo "===================================================================="
echo "[gpt-5.4] END $(date -Iseconds)"
echo "===================================================================="

echo
