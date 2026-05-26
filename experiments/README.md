# experiments/

Runners that produce the per-cell `caveman_*.jsonl` generations.

| Script | When to use it |
|---|---|
| `run_experiment.py` | Local HuggingFace models (Qwen2.5-VL-7B, Qwen3.5-9B, DeepSeek-R1-Distill-Qwen-7B, Gemma-4-E4B). Loads weights once and sweeps L0-L4 in one process. |
| `run_experiment_api.py` | API models (`gpt-*`, `claude-*`, Azure deployments tagged via `AZURE_<TAG>_API_KEY`). Reads keys from a `.env` at the repository root. |

Both runners share the same per-record schema and accept the same
`--dataset`, `--condition`, `--level`, `--output_dir` flags. See the
module docstrings for the full CLI.

The shared logic, constraint prompts, dataset loading, input compression,
answer extraction, lives in [`../src/`](../src/) and is imported via a
`sys.path` bootstrap at the top of each runner.

## Open-weight model snapshots

The four open-weight models in the paper use these HuggingFace IDs:

| Tag | HuggingFace ID |
|---|---|
| `qwen-2.5` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `qwen-3.5` | `Qwen/Qwen3.5-9B-Instruct` |
| `deepseek-r1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| `gemma-4` | `google/gemma-4-e4b-it` |

Point `run_experiment.py` at a local snapshot of any of these with
`--model_path /path/to/local/weights`. The decoder budget is
`max_new_tokens = {400, 300, 200, 150, 20}` for L0..L4, applied
identically across both channels.
