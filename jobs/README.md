# jobs/

Shell scripts that orchestrated the paper runs. Paths inside resolve
relative to the repository root, so the scripts work wherever you
clone the repo. Each one assumes you have a Python environment with
[`../requirements.txt`](../requirements.txt) installed and activated
before launching.

| Script | Model |
|---|---|
| `run_gpt-4o.sh` | OpenAI GPT-4o |
| `run_gpt-5.4.sh` | OpenAI GPT-5.4 |
| `run_claude-haiku-4-5.sh` | Anthropic Claude Haiku 4.5 |
| `run_claude-sonnet-4-6.sh` | Anthropic Claude Sonnet 4.6 |
| `run_kimi-k2.6.sh` | Moonshot Kimi-K2.6 (via Azure OpenAI client) |

Each API runner reads its key from a `.env` file at the repository
root. Designed to be launched inside `tmux`. Skips any (dataset,
condition) whose five levels are already at the expected record
count, so reruns are safe and incremental.

The two SLURM scripts target local-GPU runs of the open-weight panel
(Qwen2.5-VL-7B, Qwen3.5-9B, DeepSeek-R1-Distill-Qwen-7B, Gemma-4-E4B):

| Script | Purpose |
|---|---|
| `submit_caveman.sh` | Full L0..L4 sweep, self-chains until `accuracy_summary.json` appears so a long run survives walltime limits. |
| `submit_single_level.sh` | Single level of a single (model, dataset, condition) cell. Useful for retries. |

The SLURM `#SBATCH` partition and time limits in those scripts are
site-specific and will need adjustment for your cluster.
