#!/bin/bash
# Run Kimi-K2.6 (Azure OpenAI deployment) on all 5 datasets × 2 conditions.
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
MODEL_TAG="kimi-k2.6"
MODEL_ID="kimi-k2.6"

mkdir -p "$LOGS_DIR"

# ---- env check ----
# Assumes you have activated a Python environment with the dependencies
# from requirements.txt installed (see ../requirements.txt at the repo root).

# ---- load .env and verify Azure Kimi key ----
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "[ERROR] $REPO_ROOT/.env not found." >&2
    echo "        Add a line: AZURE_KIMI_API_KEY=<your-key>" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/.env"
set +a
if [ -z "${AZURE_KIMI_API_KEY:-}" ] || [[ "$AZURE_KIMI_API_KEY" == your_* ]]; then
    echo "[ERROR] AZURE_KIMI_API_KEY not set or still placeholder in $REPO_ROOT/.env" >&2
    echo "        Add or update the line: AZURE_KIMI_API_KEY=<your-key>" >&2
    exit 1
fi
echo "[kimi-k2.6] AZURE_KIMI_API_KEY loaded (length: ${#AZURE_KIMI_API_KEY})"

# ---- expected per-dataset counts (used for completion check) ----
declare -A EXPECTED=(
    [gsm8k]=1319
    [boolq]=3270
    [arc_easy]=2376
    [commonsenseqa]=1221
    [mmlu_stem]=3279
)

DATASETS=(gsm8k boolq arc_easy commonsenseqa mmlu_stem)
CONDITIONS=(output input)
LEVELS=(L0 L1 L2 L3 L4)

# Allow chunked parallel launches: set DATASETS_OVERRIDE to a space-separated
# list of dataset names to restrict this run to a subset.
# Example: DATASETS_OVERRIDE="gsm8k boolq" bash run_kimi_all.sh
if [ -n "${DATASETS_OVERRIDE:-}" ]; then
    # shellcheck disable=SC2206
    DATASETS=($DATASETS_OVERRIDE)
    echo "[kimi-k2.6] DATASETS_OVERRIDE set -> using only: ${DATASETS[*]}"
fi

echo "===================================================================="
echo "[kimi-k2.6] START $(date -Iseconds) on $(hostname)"
echo "  model:      $MODEL_ID (tag: $MODEL_TAG, Azure deployment: Kimi-K2.6)"
echo "  endpoint:   ${AZURE_KIMI_ENDPOINT:-<set AZURE_KIMI_ENDPOINT in .env>}"
echo "  datasets:   ${DATASETS[*]}"
echo "  conditions: ${CONDITIONS[*]}"
echo "  log dir:    $LOGS_DIR"
echo "===================================================================="

for DATASET in "${DATASETS[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        OUT_DIR="$REPO_ROOT/results/${MODEL_TAG}_${CONDITION}/${DATASET}"
        LOG="$LOGS_DIR/kimi_${DATASET}_${CONDITION}.log"

        echo
        echo "----------------------------------------------------------------"
        echo "[kimi-k2.6] $(date -Iseconds)  ${DATASET} / ${CONDITION}"
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
        # Tee stdout/stderr to per-(dataset,condition) log; -a so reruns append.
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
echo "[kimi-k2.6] END $(date -Iseconds)"
echo "===================================================================="

# Grand cost summary
echo
