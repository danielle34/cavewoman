#!/bin/bash
#
# CAVEWOMAN single-constraint-level run. No auto-chain; one walltime budget.
# If the job times out, just re-invoke; run_experiment.py resumes from the
# existing JSONL.
#
# The SBATCH directives below are defaults that need to be adjusted to your
# cluster's scheduler policy. Common things to change:
#   --partition   : your cluster's GPU partition name (e.g. gpu, gpu-shared,
#                   volta, a100). Run `sinfo` to see what's available.
#   --gres        : on some clusters you must also specify the GPU type,
#                   e.g. gpu:a100:1 or gpu:v100:1.
#   --time        : walltime cap, in HH:MM:SS. A single level over 1319
#                   GSM8K items at ~10 s/item is ~3.7 h, so 6 h is comfortable.
#   --cpus-per-task, --mem : tune to your node specs. 48G memory is enough
#                   for a 7-9B model loaded in bfloat16 on a single GPU.
#
# Launch:  bash submit_single_level.sh L1
# Monitor: squeue -u $USER    (or: tail -f results/job_log.csv)
#
# Dual-mode script:
#   - Run as `bash submit_single_level.sh L1`: re-execs itself via sbatch
#     with CAVEWOMAN_LEVEL=L1 exported.
#   - Run inside SLURM: reads $CAVEWOMAN_LEVEL and does the work.
#
#SBATCH --job-name=cavewoman_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=./logs/cavewoman_single_%j.out
#SBATCH --error=./logs/cavewoman_single_%j.err

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
LOGS_DIR="$REPO_ROOT/logs"
LOG_CSV="$RESULTS_DIR/job_log.csv"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# ---------- launcher mode ----------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    LEVEL="${1:?Usage: bash $0 <L0|L1|L2|L3|L4>}"
    case "$LEVEL" in
        L0|L1|L2|L3|L4) ;;
        *) echo "Bad level '$LEVEL' (expected L0..L4)" >&2; exit 2 ;;
    esac
    echo "[launcher] submitting single-level run for $LEVEL ..."
    exec sbatch \
        --export=ALL,CAVEWOMAN_LEVEL="$LEVEL" \
        --job-name="caveman_${LEVEL}" \
        --output="$LOGS_DIR/caveman_${LEVEL}_%j.out" \
        --error="$LOGS_DIR/caveman_${LEVEL}_%j.err" \
        "$0"
fi

# ---------- inside SLURM from here on ----------
LEVEL="${CAVEWOMAN_LEVEL:?CAVEWOMAN_LEVEL not set; was this invoked through the launcher?}"
ACTIVE_FILE="$RESULTS_DIR/.active_${LEVEL}"
OUT_DIR=""

# ---------- CSV logger ----------
log_event() {
    local ts event n_records exit_code message job_id
    ts=$(date -Iseconds)
    event="${1:-}"
    n_records="${2:-}"
    exit_code="${3:-}"
    message="${4:-}"
    job_id="${SLURM_JOB_ID:-local}"
    message=$(printf '%s' "$message" | sed 's/"/""/g')
    if [ ! -f "$LOG_CSV" ]; then
        echo "timestamp,job_id,event,level,n_records,exit_code,run_dir,message" > "$LOG_CSV"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
        "$ts" "$job_id" "$event" "$LEVEL" "$n_records" "$exit_code" "$OUT_DIR" "$message" \
        >> "$LOG_CSV"
}

# ---------- pick / reuse run dir ----------
if [ -f "$ACTIVE_FILE" ]; then
    OUT_DIR=$(cat "$ACTIVE_FILE")
    if [ -f "$OUT_DIR/accuracy_summary.json" ]; then
        log_event "skip" "" "" "$LEVEL already complete at $OUT_DIR"
        echo "[single] $OUT_DIR is already complete; clearing .active_$LEVEL and exiting."
        rm -f "$ACTIVE_FILE"
        exit 0
    fi
    log_event "resume" "" "" "Continuing $OUT_DIR"
else
    OUT_DIR="$RESULTS_DIR/single_${LEVEL}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUT_DIR"
    echo "$OUT_DIR" > "$ACTIVE_FILE"
    log_event "new_run" "" "" "Started single-level $LEVEL run at $OUT_DIR"
fi

# ---------- header ----------
echo "===================================================================="
echo "CAVEWOMAN single-level $LEVEL, job start"
echo "Time:   $(date -Iseconds)"
echo "Host:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Out:    $OUT_DIR"
echo "===================================================================="
nvidia-smi || true
log_event "job_start" "" "" "Single-level $LEVEL on $(hostname)"

# ---------- env ----------
# Assumes you have activated a Python environment with the dependencies
# from requirements.txt installed (see ../requirements.txt at the repo root).

# Unbuffered Python so progress lines hit the SLURM .out file in real time.
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

# ---------- run ----------
echo "[run] python experiments/run_experiment.py --level $LEVEL --split test --output $OUT_DIR"
python experiments/run_experiment.py --level "$LEVEL" --split test --output "$OUT_DIR"
EXIT_CODE=$?
log_event "python_exit" "" "$EXIT_CODE" "exit_code=$EXIT_CODE"

# ---------- snapshot ----------
f="$OUT_DIR/caveman_gsm8k_$LEVEL.jsonl"
if [ -f "$f" ]; then
    n=$(wc -l < "$f" | tr -d ' ')
    log_event "level_progress" "$n" "" "After this job"
fi

if [ -f "$OUT_DIR/accuracy_summary.json" ]; then
    log_event "complete" "" "" "Level $LEVEL done"
    rm -f "$ACTIVE_FILE"
    echo "[single] Level $LEVEL complete; cleared .active_$LEVEL."
else
    echo "[single] Level $LEVEL incomplete after 6h. Re-run 'bash submit_single_level.sh $LEVEL' to resume."
fi

log_event "job_end" "" "" ""
echo "===================================================================="
echo "CAVEWOMAN single-level $LEVEL, job end at $(date -Iseconds)"
echo "===================================================================="
