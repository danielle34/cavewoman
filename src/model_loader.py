"""Qwen2.5-VL-7B loader and text-only inference helper for CAVEWOMAN."""

import os
import time
from typing import Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

# Lazy import for Qwen3.5 (only available in newer transformers builds).
try:
    from transformers import Qwen3_5ForConditionalGeneration  # type: ignore  # noqa: F401
    _HAVE_QWEN35 = True
except ImportError:
    _HAVE_QWEN35 = False


# Per-family overrides. Detected from the model_path basename or the explicit
# model_type. Keep this short, only quirks that break under defaults.
#
#, NO_SYSTEM_ROLE_FAMILIES: chat templates that reject {"role":"system",...}.
#   For these we fold the system prompt into the first user turn so the same
#   CAVEWOMAN system_prompt text still reaches the model.
#, EAGER_ATTN_FAMILIES: models whose logit soft-capping silently degrades
#   under SDPA/FlashAttention2; must pass attn_implementation="eager".
NO_SYSTEM_ROLE_FAMILIES = {"gemma", "deepseek"}
EAGER_ATTN_FAMILIES = {"gemma"}


def _infer_family(model_type_hint: str, path: str) -> str:
    """Return a short family tag for routing.

    If model_type_hint is something concrete (e.g. CAVEWOMAN's --model_name
    passes 'qwen', 'qwen3.5', 'llama', 'deepseek', 'gemma', 'mistral'),
    trust it. Otherwise sniff from the path basename.
    """
    hint = (model_type_hint or "").lower()
    known = {"qwen", "qwen3.5", "llama", "deepseek", "gemma", "mistral"}
    if hint in known:
        # Special cases:
        # , short tag 'qwen' currently means Qwen2.5-VL (multimodal)
        # , short tag 'qwen3.5' means Qwen3.5-9B (uses its own VL class)
        if hint == "qwen":
            return "qwen-vl"
        if hint == "qwen3.5":
            return "qwen35-vl"
        return hint
    name = os.path.basename(os.path.normpath(path)).lower()
    if "qwen3.5" in name or "qwen35" in name:
        return "qwen35-vl"
    if "qwen" in name and "vl" in name:
        return "qwen-vl"
    if "llama" in name:
        return "llama"
    if "deepseek" in name:
        return "deepseek"
    if "gemma" in name:
        return "gemma"
    if "mistral" in name:
        return "mistral"
    return "auto"


def build_messages_for_family(family: str, system_prompt: str, user_message: str) -> list:
    """Return a chat-template message list that the model's tokenizer will accept.

    For families whose templates reject role="system", fold the system text
    into the first user turn (separated by a blank line) so the full
    CAVEWOMAN system_prompt still reaches the model.
    """
    if family in NO_SYSTEM_ROLE_FAMILIES:
        if system_prompt:
            combined = f"{system_prompt}\n\n{user_message}"
        else:
            combined = user_message
        return [{"role": "user", "content": combined}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def _format_bytes(n: int) -> str:
    gb = n / (1024 ** 3)
    return f"{gb:.2f} GiB"


def _log_gpu_memory() -> None:
    """Print per-GPU allocated/reserved memory after a load."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            name = torch.cuda.get_device_name(i)
            print(
                f"[model_loader] GPU{i} ({name}): "
                f"allocated={_format_bytes(allocated)}  reserved={_format_bytes(reserved)}"
            )
    else:
        print("[model_loader] WARNING: CUDA not available; model is on CPU.")


def load_model(model_name_or_path: str, model_type: str = "auto"):
    """Generic loader: routes between Qwen2.5-VL and AutoModelForCausalLM,
    with per-family overrides for the chat template and attention kernel.

    Routing:
    , family in {'qwen-vl'}        -> Qwen2_5_VLForConditionalGeneration
    , family in {'llama','deepseek','gemma','mistral','auto'} -> AutoModelForCausalLM
    , family in EAGER_ATTN_FAMILIES (gemma) -> attn_implementation='eager'

    All branches use torch_dtype=bfloat16, device_map='auto', and
    trust_remote_code=True so custom modeling files in the snapshot load.

    The detected family is stamped onto the returned model as
    `model._caveman_family`. `run_inference` reads that to decide whether the
    chat template needs the system-role-fold (Gemma, DeepSeek).
    """
    path = os.path.expanduser(model_name_or_path)
    family = _infer_family(model_type, path)
    is_qwen_vl = family == "qwen-vl"

    print(f"[model_loader] Loading tokenizer from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if family in EAGER_ATTN_FAMILIES:
        load_kwargs["attn_implementation"] = "eager"

    if is_qwen_vl:
        model_cls = Qwen2_5_VLForConditionalGeneration
        cls_name = "Qwen2_5_VLForConditionalGeneration"
    elif family == "qwen35-vl":
        if not _HAVE_QWEN35:
            raise RuntimeError(
                "Qwen3_5ForConditionalGeneration not importable from transformers. "
                "Upgrade transformers to a version that includes Qwen3.5 support."
            )
        from transformers import Qwen3_5ForConditionalGeneration  # type: ignore  # noqa: PLC0415
        model_cls = Qwen3_5ForConditionalGeneration
        cls_name = "Qwen3_5ForConditionalGeneration"
    else:
        model_cls = AutoModelForCausalLM
        cls_name = "AutoModelForCausalLM"
    print(f"[model_loader] family={family}  cls={cls_name}  "
          f"extra_kwargs={ {k:v for k,v in load_kwargs.items() if k not in {'torch_dtype','device_map','trust_remote_code'}} }")

    t0 = time.time()
    model = model_cls.from_pretrained(path, **load_kwargs)
    model.eval()
    model._caveman_family = family  # noqa: SLF001, read by run_inference()
    print(f"[model_loader] Model loaded in {time.time(), t0:.1f}s")
    _log_gpu_memory()
    return tokenizer, model


def load_qwen(model_path: str):
    """Load a Qwen2.5-VL snapshot from `model_path` (local dir or HuggingFace ID)."""
    return load_model(model_path, model_type="qwen")


@torch.inference_mode()
def run_inference(
    tokenizer,
    model,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> Dict:
    """Run a single chat-formatted generation; return decoded text + metrics.

    The message structure is family-aware via `model._caveman_family`:
    families whose chat template rejects role='system' (Gemma, DeepSeek) get
    the system prompt folded into the first user turn. See
    `build_messages_for_family()`.
    """
    family = getattr(model, "_caveman_family", "auto")
    messages = build_messages_for_family(family, system_prompt, user_message)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    t0 = time.time()
    output_ids = model.generate(**inputs, **gen_kwargs)
    latency_s = time.time(), t0

    new_tokens = output_ids[0][input_token_count:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "output": output_text,
        "input_tokens": int(input_token_count),
        "output_tokens": int(new_tokens.shape[0]),
        "latency_s": float(latency_s),
    }


if __name__ == "__main__":
    # Smoke test: load the model at CAVEWOMAN_MODEL_PATH and ask a trivial
    # arithmetic question under L0.
    model_path = os.environ.get("CAVEWOMAN_MODEL_PATH")
    if not model_path:
        import sys as _sys
        print("[smoke] CAVEWOMAN_MODEL_PATH not set. Export it to a local "
              "model snapshot path or a HuggingFace model ID "
              "(e.g. Qwen/Qwen2.5-VL-7B-Instruct), then re-run.",
              file=_sys.stderr)
        raise SystemExit(2)

    try:
        from constraint_prompts import CONSTRAINT_PROMPTS, get_max_tokens
        system_prompt = CONSTRAINT_PROMPTS["L0"]
        max_new = get_max_tokens("L0")
    except ImportError:
        print("[smoke] constraint_prompts.py not importable; using fallback prompt.")
        system_prompt = (
            "Solve the math problem step by step. End with a line "
            "'Answer: <number>'."
        )
        max_new = 200

    tokenizer, model = load_model(model_path, model_type="auto")

    print("\n[smoke] Running inference: 'What is 2 + 2?'")
    result = run_inference(
        tokenizer=tokenizer,
        model=model,
        system_prompt=system_prompt,
        user_message="What is 2 + 2?",
        max_new_tokens=max_new,
        temperature=0.0,
    )
    print("-" * 60)
    print("OUTPUT:")
    print(result["output"])
    print("-" * 60)
    print(f"input_tokens  : {result['input_tokens']}")
    print(f"output_tokens : {result['output_tokens']}")
    print(f"latency_s     : {result['latency_s']:.2f}")
