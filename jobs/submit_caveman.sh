#!/bin/bash
#
# CAVEWOMAN full L0..L4 sweep, self-chains until accuracy_summary.json appears.
#
# The SBATCH directives below are defaults that need to be adjusted to your
# cluster's scheduler policy. Common things to change:
#   --partition   : your cluster's GPU partition name (e.g. gpu, gpu-shared,
#                   volta, a100). Run `sinfo` to see what's available.
#   --gres        : on some clusters you must also specify the GPU type,
#                   e.g. gpu:a100:1 or gpu:v100:1.
#   --time        : walltime cap, in HH:MM:SS. Cluster policies vary widely.
#   --cpus-per-task, --mem : tune to your node specs. 48G memory is enough
#                   for a 7-9B model loaded in bfloat16 on a single GPU.
#
# Submit:   sbatch submit_caveman.sh
# Monitor:  squeue -u $USER    (or: tail -f results/job_log.csv)
#
#SBATCH --job-name=cavewoman_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=./logs/cavewoman_%j.out
#SBATCH --error=./logs/cavewoman_%j.err

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
LOGS_DIR="$REPO_ROOT/logs"
LOG_CSV="$RESULTS_DIR/job_log.csv"
ACTIVE_FILE="$RESULTS_DIR/.active_run"
MAX_CHAINS=10
OUT_DIR=""

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# ---------- CSV logger ----------
# usage: log_event EVENT [LEVEL] [N_RECORDS] [EXIT_CODE] [MESSAGE]
log_event() {
    local ts event level n_records exit_code message job_id
    ts=$(date -Iseconds)
    event="${1:-}"
    level="${2:-}"
    n_records="${3:-}"
    exit_code="${4:-}"
    message="${5:-}"
    job_id="${SLURM_JOB_ID:-local}"
    # Double any embedded " for CSV quoting.
    message=$(printf '%s' "$message" | sed 's/"/""/g')
    if [ ! -f "$LOG_CSV" ]; then
        echo "timestamp,job_id,event,level,n_records,exit_code,run_dir,message" > "$LOG_CSV"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
        "$ts" "$job_id" "$event" "$level" "$n_records" "$exit_code" "${OUT_DIR}" "$message" \
        >> "$LOG_CSV"
}

# ---------- header ----------
echo "===================================================================="
echo "CAVEWOMAN full sweep, job start"
echo "Time:    $(date -Iseconds)"
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-<not in slurm>}"
echo "===================================================================="
nvidia-smi || true
echo

# ---------- resume-or-new run dir ----------
if [ -f "$ACTIVE_FILE" ]; then
    OUT_DIR=$(cat "$ACTIVE_FILE")
    if [ -f "$OUT_DIR/accuracy_summary.json" ]; then
        log_event "skip" "" "" "" "Run already complete: $OUT_DIR"
        echo "[chain] $OUT_DIR is already complete; clearing .active_run and exiting."
        rm -f "$ACTIVE_FILE"
        exit 0
    fi
    log_event "resume" "" "" "" "Continuing $OUT_DIR"
    echo "[chain] Continuing run: $OUT_DIR"
else
    OUT_DIR="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUT_DIR"
    echo "$OUT_DIR" > "$ACTIVE_FILE"
    log_event "new_run" "" "" "" "Started new run $OUT_DIR"
    echo "[chain] New run dir: $OUT_DIR"
fi

# ---------- chain counter ----------
CHAIN_FILE="$OUT_DIR/.chain_count"
CHAIN_NUM=$(( $(cat "$CHAIN_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$CHAIN_NUM" > "$CHAIN_FILE"
log_event "job_start" "" "" "" "Chain $CHAIN_NUM on $(hostname)"
echo "[chain] This is chain #$CHAIN_NUM (max $MAX_CHAINS)."

# ---------- env ----------
# Assumes you have activated a Python environment with the dependencies
# from requirements.txt installed (see ../requirements.txt at the repo root).

# Unbuffered Python so progress lines hit the SLURM .out file in real time.
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

# ---------- run ----------
echo "[run] python experiments/run_experiment.py --level all --split test --output $OUT_DIR"
python experiments/run_experiment.py --level all --split test --output "$OUT_DIR"
EXIT_CODE=$?
log_event "python_exit" "" "" "$EXIT_CODE" "Chain $CHAIN_NUM exit_code=$EXIT_CODE"

# ---------- per-level snapshot ----------
for lvl in L0 L1 L2 L3 L4; do
    f="$OUT_DIR/caveman_gsm8k_$lvl.jsonl"
    if [ -f "$f" ]; then
        n=$(wc -l < "$f" | tr -d ' ')
        log_event "level_progress" "$lvl" "$n" "" "After chain $CHAIN_NUM"
    fi
done

# ---------- decide: complete, resubmit, or give up ----------
if [ -f "$OUT_DIR/accuracy_summary.json" ]; then
    log_event "complete" "" "" "" "All levels done after $CHAIN_NUM chain(s)"
    rm -f "$ACTIVE_FILE"
    echo "[chain] Run complete after $CHAIN_NUM chain(s). Cleared .active_run."
elif [ "$CHAIN_NUM" -lt "$MAX_CHAINS" ]; then
    NEXT_JOB=$(sbatch --parsable "$REPO_ROOT/submit_caveman.sh" 2>&1 || echo "SUBMIT_FAILED")
    log_event "resubmit" "" "" "" "next_job_id=$NEXT_JOB chain=$((CHAIN_NUM+1))/$MAX_CHAINS"
    echo "[chain] Run incomplete; resubmitted as job $NEXT_JOB"
else
    log_event "chain_exhausted" "" "" "" "Reached MAX_CHAINS=$MAX_CHAINS without completing"
    echo "[chain] MAX_CHAINS=$MAX_CHAINS reached; will not auto-resubmit. Investigate."
fi

log_event "job_end" "" "" "" ""
echo "===================================================================="
echo "CAVEWOMAN full sweep, job end at $(date -Iseconds)"
echo "===================================================================="
