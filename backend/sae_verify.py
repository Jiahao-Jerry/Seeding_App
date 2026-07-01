"""
SAE-based rewrite verification.

Pipeline:
  text → Qwen2.5-7B layer 24 (mean-pool) → SAE encoder (top-k=25) → 128-dim acts
  Δacts = acts(Post') − acts(Post)   [Post uses pre-computed corpus acts if available]
  axis_shift[ax] = Δacts · ridge_coef[ax]   [intercept cancels in differences]

Verdict: transform is "clean" if target axes dominate the shift.
"""

import numpy as np
import torch
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SAE_PATH = DATA_DIR / "sae2/variants/qwen24_knn_k25_l0004/sae_model.pt"
RIDGE_PATH = DATA_DIR / "sae_ridge_models.npz"
CORPUS_ACTS_PATH = DATA_DIR / "sae_activations.npy"
CORPUS_PARQUET = DATA_DIR / "annotated_posts.parquet"

AXIS_NAMES = [
    "reading_level", "concreteness", "narrativity", "hedging",
    "tone", "warmth", "self_disclosure", "casualness", "humor",
]
KNN_K = 25
QWEN_LAYER = 24
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B"
QWEN_SCALE = 85.62510681152344   # mean L2 norm of training hidden states

# ── Singletons ────────────────────────────────────────────────────
_sae: dict | None = None
_ridge: dict | None = None
_corpus_acts: np.ndarray | None = None
_corpus_pid_to_row: dict | None = None
_qwen_model = None
_qwen_tok = None
_device: torch.device | None = None


# ── Device ───────────────────────────────────────────────────────

def _get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


# ── SAE + Ridge + corpus acts (light, load eagerly) ──────────────

def load_sae():
    global _sae, _ridge, _corpus_acts, _corpus_pid_to_row
    if _sae is not None:
        return

    ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    _sae = {
        "W_enc": sd["W_enc"].numpy().astype(np.float32),  # (128, 3584)
        "b_enc": sd["b_enc"].numpy().astype(np.float32),  # (128,)
    }

    npz = np.load(RIDGE_PATH)
    _ridge = {ax: npz[f"{ax}_coef"].astype(np.float32) for ax in AXIS_NAMES}

    _corpus_acts = np.load(CORPUS_ACTS_PATH).astype(np.float32)
    import pandas as pd
    df = pd.read_parquet(CORPUS_PARQUET, columns=["post_id"])
    _corpus_pid_to_row = {str(p): i for i, p in enumerate(df["post_id"].astype(str))}


# ── Qwen (heavy, load lazily on first verify call) ───────────────

def load_qwen():
    global _qwen_model, _qwen_tok
    if _qwen_model is not None:
        return

    from transformers import AutoTokenizer, AutoModel
    device = _get_device()
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32

    print(f"[sae_verify] Loading {QWEN_MODEL_ID} on {device} ({dtype}) …")
    _qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    _qwen_model = AutoModel.from_pretrained(QWEN_MODEL_ID, torch_dtype=dtype)
    _qwen_model = _qwen_model.to(device)
    _qwen_model.eval()
    print(f"[sae_verify] Qwen loaded.")


# ── Core encode ───────────────────────────────────────────────────

def _get_hidden(text: str) -> np.ndarray:
    """Mean-pool Qwen2.5-7B layer-24 hidden states over tokens."""
    load_qwen()
    device = _get_device()

    inputs = _qwen_tok(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        out = _qwen_model(**inputs, output_hidden_states=True)

    h = out.hidden_states[QWEN_LAYER]   # (1, seq_len, 3584)
    hidden = h[0].mean(dim=0).float().cpu().numpy()  # (3584,)
    return hidden


def _encode_sae(hidden: np.ndarray) -> np.ndarray:
    """
    SAE encoder: normalize input → linear → top-k sparsity.
    Scale matches training normalization (divide by mean L2 norm).
    """
    load_sae()
    h_norm = hidden / QWEN_SCALE
    pre_acts = _sae["W_enc"] @ h_norm + _sae["b_enc"]   # (128,)
    acts = np.maximum(0.0, pre_acts)                      # ReLU

    # Top-k sparsity: keep only top KNN_K activations
    if KNN_K < len(acts) and acts.max() > 0:
        threshold = np.partition(acts, -KNN_K)[-KNN_K]
        acts = np.where(acts >= threshold, acts, 0.0)

    return acts.astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    """Full pipeline: text → 128-dim SAE activations."""
    return _encode_sae(_get_hidden(text))


# ── Axis shift projection ─────────────────────────────────────────

def axis_shifts(delta_acts: np.ndarray) -> dict[str, float]:
    """
    Project Δacts through Ridge weight vectors → per-axis shift.
    Intercept cancels when comparing two posts, so only dot-products needed.
    """
    load_sae()
    return {ax: float(delta_acts @ _ridge[ax]) for ax in AXIS_NAMES}


# ── Main verify function ──────────────────────────────────────────

def verify_rewrite(
    original_text: str,
    rewritten_text: str,
    target_axes: list[str],
    orig_post_id: str | None = None,
) -> dict:
    """
    SAE verification of a rewrite.

    Returns axis shifts, top changed features, and a clean/leaked verdict.
    If orig_post_id matches a corpus post, uses pre-computed activations
    (saves one Qwen forward pass).
    """
    load_sae()

    # Original activations
    if orig_post_id is not None and _corpus_pid_to_row is not None:
        row = _corpus_pid_to_row.get(str(orig_post_id))
        acts_orig = _corpus_acts[row] if row is not None else _encode_sae(_get_hidden(original_text))
    else:
        acts_orig = _encode_sae(_get_hidden(original_text))

    # Rewritten activations — always run Qwen
    acts_new = _encode_sae(_get_hidden(rewritten_text))

    delta = acts_new - acts_orig                          # (128,)
    shifts = axis_shifts(delta)

    # Top 20 most-changed features
    top_idxs = np.argsort(np.abs(delta))[::-1][:20]
    top_features = [
        {
            "feature": int(i),
            "orig": round(float(acts_orig[i]), 4),
            "new": round(float(acts_new[i]), 4),
            "delta": round(float(delta[i]), 4),
        }
        for i in top_idxs
    ]

    # Verdict: target axis shifts should dominate unintended shifts
    target_mag = max((abs(shifts[ax]) for ax in target_axes if ax in shifts), default=0.0)
    other_mag  = max((abs(v) for ax, v in shifts.items() if ax not in target_axes), default=0.0)

    # Clean: target shift is meaningful AND at least 1.5× any unintended shift
    passed = target_mag > 0.03 and (other_mag == 0.0 or target_mag / other_mag > 1.5)

    return {
        "axis_shifts":       {ax: round(v, 4) for ax, v in shifts.items()},
        "target_axes":       target_axes,
        "acts_orig":         acts_orig.tolist(),
        "acts_new":          acts_new.tolist(),
        "delta_acts":        delta.tolist(),
        "top_feature_changes": top_features,
        "passed":            passed,
        "verdict":           "clean" if passed else "leaked",
        "max_target_shift":  round(target_mag, 4),
        "max_other_shift":   round(other_mag, 4),
    }
