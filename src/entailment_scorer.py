"""Local NLI-based entailment scoring for CAVEWOMAN semantic preservation.

Uses the `cross-encoder/nli-deberta-v3-base` model via sentence_transformers.
Cross-encoder takes (premise, hypothesis) pairs and returns logits over
{contradiction, entailment, neutral}; we softmax them into probabilities.

Conventions used by the rest of the pipeline:
    lx_entails_l0   : premise = Lx output (constrained), hypothesis = L0 (free)
    l0_entails_lx   : premise = L0 output (free),       hypothesis = Lx (constrained)
    bidirectional   : both directions land on 'entailment'

No OpenAI / no remote API. The first call downloads ~180 MB into the
sentence_transformers cache (~/.cache/torch/sentence_transformers/...).
"""

from typing import Dict, List, Tuple

import numpy as np


NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
DEFAULT_BATCH_SIZE = 32


def load_nli_model():
    """Load the cross-encoder NLI model and report its size."""
    from sentence_transformers import CrossEncoder
    print(f"[nli] Loading {NLI_MODEL_NAME} ...")
    model = CrossEncoder(NLI_MODEL_NAME)
    # Best-effort parameter count.
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"[nli] Loaded: ~{n_params / 1e6:.1f}M parameters")
    except Exception:
        print("[nli] Loaded.")
    return model


def _label_map(model) -> List[str]:
    """Return ['contradiction', 'entailment', 'neutral']-style list in index order."""
    id2label = getattr(getattr(model, "model", model), "config", None)
    id2label = getattr(id2label, "id2label", None) if id2label is not None else None
    if not id2label:
        # Known default for cross-encoder/nli-deberta-v3-base
        return ["contradiction", "entailment", "neutral"]
    keys = sorted(int(k) for k in id2label.keys())
    return [str(id2label[k]).lower() for k in keys]


def _softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x, x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _row_to_result(probs_row: np.ndarray, labels: List[str]) -> Dict:
    by_label = {labels[i]: float(probs_row[i]) for i in range(len(labels))}
    top = max(by_label, key=by_label.get)
    return {
        "label": top,
        "entailment_prob": float(by_label.get("entailment", 0.0)),
        "contradiction_prob": float(by_label.get("contradiction", 0.0)),
        "neutral_prob": float(by_label.get("neutral", 0.0)),
    }


def score_entailment(nli_model, premise: str, hypothesis: str) -> Dict:
    """Score a single (premise, hypothesis) pair.

    Returns {label, entailment_prob, contradiction_prob, neutral_prob}.
    """
    logits = nli_model.predict([(premise, hypothesis)])
    arr = np.asarray(logits)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    probs = _softmax(arr)[0]
    return _row_to_result(probs, _label_map(nli_model))


def batch_score_entailment(
    nli_model,
    pairs: List[Tuple[str, str]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict]:
    """Batch-score (premise, hypothesis) pairs in chunks of `batch_size`."""
    if not pairs:
        return []
    logits = nli_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    arr = np.asarray(logits)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    probs = _softmax(arr)
    labels = _label_map(nli_model)
    return [_row_to_result(probs[i], labels) for i in range(probs.shape[0])]


def compute_semantic_preservation(
    nli_model,
    l0_output: str,
    lx_output: str,
) -> Dict:
    """Bidirectional entailment between an L0 (free) and Lx (constrained) output.

    Returns:
        {
          "bidirectional_entailment": bool,
          "lx_entails_l0": {label, entailment_prob, contradiction_prob, neutral_prob},
          "l0_entails_lx": {label, ...},
        }
    """
    pairs = [
        (lx_output or "", l0_output or ""),   # premise=Lx, hypothesis=L0 -> "Lx entails L0"
        (l0_output or "", lx_output or ""),   # premise=L0, hypothesis=Lx -> "L0 entails Lx"
    ]
    lx_to_l0, l0_to_lx = batch_score_entailment(nli_model, pairs, batch_size=2)
    bidirectional = (lx_to_l0["label"] == "entailment") and (l0_to_lx["label"] == "entailment")
    return {
        "bidirectional_entailment": bool(bidirectional),
        "lx_entails_l0": lx_to_l0,
        "l0_entails_lx": l0_to_lx,
    }


if __name__ == "__main__":
    nli = load_nli_model()
    print("\n[demo] single-pair score:")
    print(score_entailment(
        nli,
        premise="Janet earned 18 dollars by selling 9 eggs at 2 dollars each.",
        hypothesis="She made eighteen dollars.",
    ))

    print("\n[demo] bidirectional CAVEWOMAN-style comparison:")
    sp = compute_semantic_preservation(
        nli,
        l0_output=(
            "Janet starts with 16 eggs. She eats 3 for breakfast (16, 3 = 13) "
            "and uses 4 for muffins (13, 4 = 9). She sells 9 eggs at $2 each, "
            "so she earns $18. Answer: 18"
        ),
        lx_output="Eggs 16. Eaten 3. Baked 4. Remain 9. Price 2. Earn 18. Answer: 18",
    )
    print(f"  bidirectional: {sp['bidirectional_entailment']}")
    print(f"  Lx -> L0     : {sp['lx_entails_l0']}")
    print(f"  L0 -> Lx     : {sp['l0_entails_lx']}")

    print("\n[demo] contradiction case:")
    sp_bad = compute_semantic_preservation(
        nli,
        l0_output="Janet earned 18 dollars per day from her eggs.",
        lx_output="Janet earned 5 dollars per day from her eggs.",
    )
    print(f"  bidirectional: {sp_bad['bidirectional_entailment']}")
    print(f"  Lx -> L0     : {sp_bad['lx_entails_l0']}")
    print(f"  L0 -> Lx     : {sp_bad['l0_entails_lx']}")
