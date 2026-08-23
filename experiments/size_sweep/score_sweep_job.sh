#!/bin/bash
# Adds bidirectional-NLI and embedding-cosine sidecars to every finished sweep
# cell that lacks them. Idempotent: skips cells whose _with_entailment sidecar
# already exists.
#
# Scope a pass with SCORE_TAGS="tag1 tag2"; see the defaults below.

set -u
# Site configuration: override via environment, or edit here.
#   CAVEWOMAN_ENV_SETUP  shell snippet to activate your Python env (default: none)
#   CAVEWOMAN_ROOT       path to this repo checkout (default: script's repo root)
: "${CAVEWOMAN_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ -n "${CAVEWOMAN_ENV_SETUP:-}" ]; then eval "$CAVEWOMAN_ENV_SETUP"; fi
export PYTHONUNBUFFERED=1

echo "=== score sweep job ==="; date -Iseconds; hostname; nvidia-smi -L | head -2
cd "$CAVEWOMAN_ROOT"

TAGS=(${SCORE_TAGS:-qwen2.5-3b qwen2.5-7b qwen2.5-14b qwen2.5-32b qwen3.5-think qwen3.5-nothink})
DATASETS=(${SCORE_DATASETS:-gsm8k boolq arc_easy commonsenseqa mmlu_stem})
# Levels to score, and the level whose sidecar marks the cell already done.
# Default is the original {L0,L1,L2,L4} sweep; the L3 backfill passes
# SCORE_LEVELS="L0 L3" SCORE_MARK=L3.
SCORE_LEVELS=(${SCORE_LEVELS:-L0 L1 L2 L4})
SCORE_MARK="${SCORE_MARK:-L1}"
LEVEL_CSV=$(IFS=,; echo "${SCORE_LEVELS[*]}")

for TAG in "${TAGS[@]}"; do
  for COND in output input; do
    for DS in "${DATASETS[@]}"; do
      DIR="results/${TAG}_${COND}/${DS}"
      L0="$DIR/caveman_${TAG}_${DS}_${COND}_L0.jsonl"
      L1="$DIR/caveman_${TAG}_${DS}_${COND}_L1.jsonl"
      DONE_MARK="$DIR/caveman_${TAG}_${DS}_${COND}_${SCORE_MARK}_with_entailment.jsonl"
      # score only when the cell is COMPLETE: every requested level file present
      # with exactly N_EXPECTED records (guards against scoring a cell the
      # sweep job is still writing).
      N_EXPECTED=500
      COMPLETE=1
      for LV in "${SCORE_LEVELS[@]}"; do
        F="$DIR/caveman_${TAG}_${DS}_${COND}_${LV}.jsonl"
        if [ ! -f "$F" ] || [ "$(wc -l < "$F")" -ne "$N_EXPECTED" ]; then
          COMPLETE=0; break
        fi
      done
      [ "$COMPLETE" -eq 1 ] || continue
      if [ ! -f "$DONE_MARK" ]; then
        echo "--- entailment: $TAG $COND $DS ($(date -Iseconds)) ---"
        python evaluation/add_entailment_scores.py \
            --results_dir "$DIR" --model_name "$TAG" --dataset "$DS" \
            --condition "$COND" --levels "$LEVEL_CSV" || echo "[score] entailment FAILED $TAG $COND $DS"
        echo "--- embeddings: $TAG $COND $DS ---"
        python evaluation/add_embedding_scores.py \
            --results_dir "$DIR" --model_name "$TAG" --dataset "$DS" \
            --condition "$COND" --levels "$LEVEL_CSV" || echo "[score] embeddings FAILED $TAG $COND $DS"
      fi
    done
  done
done
echo "[score_sweep] pass complete $(date -Iseconds)"
