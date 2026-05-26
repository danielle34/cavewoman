# analysis/llmlingua/

LLMLingua-2 input-compression comparison reported in the paper appendix.
Reruns the input channel using the learned LLMLingua-2 compressor on
3 models (GPT-4o, Claude Sonnet 4.6, Qwen2.5-VL-7B) x 3 datasets
(GSM8K, BoolQ, ARC-Easy) at tau = 0.5 (LLMLingua-2 default) and
tau = 0.8 (matched to CAVEWOMAN's telegraphic retention).

Pipeline, in execution order:

| Script | Purpose |
|---|---|
| `ratios.py` | Compute per-item LLMLingua-2 compression ratios at the target tau. |
| `compress.py` | Apply LLMLingua-2 to the input questions at the chosen tau. |
| `random_compress.py` | Random-token-drop control at the same retention rate. |
| `inference_gpt4o.py` | Run GPT-4o on the LLMLingua-compressed inputs. |
| `inference_haiku.py` | Run Claude Haiku 4.5. |
| `inference_sonnet.py` | Run Claude Sonnet 4.6. |
| `inference_qwen.py` | Run Qwen2.5-VL-7B locally. |
| `inference_qwen_random.py` | Run Qwen2.5-VL-7B on the random-control compressed inputs. |
| `inference_local.py` | Generic local-HF runner (drop-in for any new open-weight model). |
| `score.py` | Accuracy + NLI scoring of LLMLingua outputs against L0. |
| `compare.py` | Build the cross-method comparison table reported in the paper. |
| `score_bertscore.py` | BERTScore on LLMLingua vs CAVEWOMAN matched outputs. |
| `finish_download_and_run.sh` | End-to-end driver for the BERTScore scoring pass. |

Reference: Pan et al., *LLMLingua-2*, 2024.
