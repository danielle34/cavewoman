#!/usr/bin/env bash
# Finishes downloading roberta-large into the local tmpfs cache and then
# kicks off the BERTScore scoring job. Designed to be run inside tmux.
set -euo pipefail

export HF_HOME=/tmp/hf_cache_bertscore
export HUGGINGFACE_HUB_CACHE=/tmp/hf_cache_bertscore
export HF_HUB_ENABLE_HF_TRANSFER=0

LOGDIR="../../logs"
mkdir -p "$LOGDIR"

echo "[$(date)] resuming roberta-large download into $HF_HOME"
python, <<'PY'
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(repo_id="roberta-large",
                      cache_dir="/tmp/hf_cache_bertscore",
                      max_workers=4)
print(f"download OK in {time.time()-t0:.1f}s -> {p}", flush=True)
PY

echo "[$(date)] launching BERTScore scoring"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python -u analysis/llmlingua/score_bertscore.py 2>&1 | tee -a "$LOGDIR/bertscore_llmlingua.log"
echo "[$(date)] BERTScore scoring finished"
