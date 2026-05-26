#!/usr/bin/env bash
# CAVEWOMAN smoke test for any SLURM cluster with a single GPU.
#
# Purpose: prove the local-GPU generation pipeline, the answer-extraction
# scorer, and the conda environment are all wired up correctly, without
# spending real GPU-hours on a full sweep.
#
# What it does:
#   1. Loads the conda env.
#   2. Runs the self-contained POS-filter demo (no model needed).
#   3. Generates 5 items from Qwen2.5-VL-7B on GSM8K, Condition A, all
#      five levels L0..L4 (small enough to finish in well under 10 min).
#   4. Prints the per-cell accuracy_summary.json so you can eyeball
#      that scoring and provenance worked end-to-end.
#
# Expected wall time:  ~3-10 minutes including model load.
# Expected VRAM:       ~16 GB for Qwen2.5-VL-7B at bfloat16.
# Expected output:     per-level JSONLs + accuracy_summary.json under
#                      $REPO_ROOT/results/smoke_test_<timestamp>/.
#
# The SBATCH directives below are placeholders. Edit them for your site,
# or override on the sbatch command line, for example:
#     sbatch --partition=<your_partition> --gres=gpu:1 \
#            scripts/smoke_test_slurm.sh
#
# Then watch:
#     squeue -u "$USER"
#     tail -f logs/smoke_test-*.out

#SBATCH --job-name=cavewoman_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_test-%j.out
#SBATCH --error=logs/smoke_test-%j.err

set -u
set -o pipefail

# ---- environment (override via env vars before sbatch) ----
# CAVEWOMAN_MODEL_PATH is required: a local model snapshot path or a
# HuggingFace ID (e.g. Qwen/Qwen2.5-VL-7B-Instruct).
if [ -z "${CAVEWOMAN_MODEL_PATH:-}" ]; then
    echo "[smoke] CAVEWOMAN_MODEL_PATH is not set." >&2
    echo "        Export it to a local model snapshot path or a HuggingFace ID, then re-submit." >&2
    exit 2
fi
CAVEWOMAN_DATASET="${CAVEWOMAN_DATASET:-gsm8k}"
CAVEWOMAN_N_ITEMS="${CAVEWOMAN_N_ITEMS:-5}"

# Assumes you have already activated a Python environment with the
# CAVEWOMAN requirements installed (see ../requirements.txt at the repo root).

# Some PyTorch builds dynamically link libcusparseLt at import time. If
# the library is shipped via a pip dep under
# <env>/lib/python*/site-packages/nvidia/cusparselt/lib but that path is
# not on LD_LIBRARY_PATH, `import torch` raises ImportError. Setting it
# inside Python (os.environ) is too late: the dynamic linker has already
# resolved torch's deps. Export it here, in the shell, before the first
# `python` invocation. Harmless if the directory does not exist.
CUSPARSELT_DIR="$(find "$CONDA_PREFIX/lib" -maxdepth 5 -type d -path '*/nvidia/cusparselt/lib' 2>/dev/null | head -1)"
if [ -n "$CUSPARSELT_DIR" ]; then
    export LD_LIBRARY_PATH="$CUSPARSELT_DIR:${LD_LIBRARY_PATH:-}"
fi

# ---- where to write ----
# SLURM copies the submitted script to /var/spool/, so $0/dirname is useless.
# $SLURM_SUBMIT_DIR is set by sbatch to the directory the job was submitted from.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.."; pwd)}"
cd "$REPO_ROOT"
SCRATCH_OUT="$REPO_ROOT/results/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SCRATCH_OUT"

echo "==== node + GPU ===="
hostname
nvidia-smi -L
echo

# ---- step 1: self-contained POS-filter demo ----
echo "==== step 1: POS-filter demo (no model load) ===="
python scripts/pos_filter_demo.py \
    --text "Janet's ducks lay 16 eggs per day. She eats three for breakfast." \
    --level L0,L1,L2,L3,L4
echo

# ---- step 2: real generation ----
echo "==== step 2: Qwen2.5-VL-7B on $CAVEWOMAN_DATASET Cond A, n=$CAVEWOMAN_N_ITEMS, all 5 levels ===="
python run_experiment.py \
    --model_name qwen-2.5 \
    --model_path "$CAVEWOMAN_MODEL_PATH" \
    --dataset   "$CAVEWOMAN_DATASET" \
    --condition input \
    --level     all \
    --n         "$CAVEWOMAN_N_ITEMS" \
    --seed      42 \
    --output    "$SCRATCH_OUT"
echo

# ---- step 3: inspect the per-cell summary ----
echo "==== step 3: per-cell summary ===="
SUMMARY="$SCRATCH_OUT/accuracy_summary.json"
if [ -f "$SUMMARY" ]; then
    python -m json.tool "$SUMMARY" | head -40
else
    echo "FAIL: $SUMMARY not produced"
    exit 1
fi

echo
echo "==== done ===="
echo "Outputs are in: $SCRATCH_OUT"
echo "If accuracy_summary.json above shows L0..L4 with n=$CAVEWOMAN_N_ITEMS, the pipeline is wired up."
