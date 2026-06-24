"""
LLM hidden-state extraction for Option 2 SAE.

For each post, runs the text through a base LLM and captures the
residual-stream activation at a chosen layer, mean-pooled across the
post's valid (non-padding) tokens.

Why base (not Instruct):
  Instruction-tuned models bias representations toward "what would I say in
  response" rather than "what does this input look like." Base models give
  cleaner stylistic encodings, which is what we want for register discovery.

Why mean-pool over tokens (not last-token):
  Last-token pooling biases toward end-of-post words. Mean-pool gives a
  position-weighted average of the whole post in the LLM's representation.

Why layer 18 of 28 by default:
  Anthropic / OpenAI SAE work consistently targets the middle-to-late
  residual stream (~60-70% depth) where abstract features (sentiment,
  register, style) live. Earlier layers carry surface features; very late
  layers are biased toward next-token prediction logits.

Output shape: (N_posts, hidden_dim), float32, aligned with df row order.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Use one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _resolve_device(requested: str) -> str:
    """Return a device string that actually works on this machine."""
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("  [warn] MPS requested but unavailable — falling back to CPU.")
            return "cpu"
        if not torch.backends.mps.is_built():
            print("  [warn] MPS not built into this torch — falling back to CPU.")
            return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA requested but unavailable — falling back to CPU.")
        return "cpu"
    return requested


def load_base_lm(
    model_name: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, list]:
    """Load tokenizer + causal LM and return (tokenizer, model, decoder_layers)."""
    print(f"  loading tokenizer for {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"  loading weights ({dtype}) — this can take a minute for first-time download…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    print(f"  weights loaded in {time.time() - t0:.1f}s")

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise RuntimeError(
            f"Cannot find model.model.layers on {model_name}. "
            "This loader assumes a Llama/Qwen/Mistral-style architecture."
        )
    return tokenizer, model, layers


def extract_activations(
    texts: list[str],
    model_name: str,
    layer_idx: int,
    max_length: int = 128,
    batch_size: int = 4,
    device: str = "mps",
    dtype_name: str = "bfloat16",
    log_every_batches: int = 25,
) -> np.ndarray:
    """
    Mean-pooled residual-stream activations at layer_idx for each text.

    Returns
    -------
    np.ndarray of shape (len(texts), hidden_dim), dtype float32.
    """
    device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name)

    tokenizer, model, layers = load_base_lm(model_name, device, dtype)

    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range; model has {len(layers)} decoder layers."
        )
    print(f"  hooking layer {layer_idx} of {len(layers)} "
          f"(~{100 * layer_idx / max(len(layers) - 1, 1):.0f}% depth)")

    captured: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        # Llama/Qwen-style decoder blocks return (hidden_states, ...) or just a tensor
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    handle = layers[layer_idx].register_forward_hook(hook)

    n = len(texts)
    all_pooled: list[np.ndarray] = []
    t_start = time.time()
    try:
        for b_idx, start in enumerate(range(0, n, batch_size)):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                _ = model(**enc, use_cache=False)

            acts = captured["acts"]               # (B, T, D), in model dtype
            mask = enc.attention_mask.unsqueeze(-1).to(acts.dtype)  # (B, T, 1)
            summed = (acts * mask).sum(dim=1)     # (B, D)
            counts = mask.sum(dim=1).clamp_min(1) # (B, 1)
            pooled = (summed / counts).float()    # (B, D), upcast to fp32 for numpy

            all_pooled.append(pooled.cpu().numpy().astype(np.float32))

            done = min(start + batch_size, n)
            if (b_idx + 1) % log_every_batches == 0 or done >= n:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                remaining = max(n - done, 0)
                eta = remaining / max(rate, 1e-6)
                print(f"  {done:4d}/{n} posts | {rate:5.2f} posts/s | "
                      f"elapsed {elapsed:5.0f}s | eta {eta:5.0f}s")
    finally:
        handle.remove()

    # Free model weights from device memory before returning — we don't need
    # them again, and unified memory is precious for the SAE training step.
    del model
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return np.vstack(all_pooled)


def activations_summary(activations: np.ndarray) -> dict:
    norms = np.linalg.norm(activations, axis=1)
    return {
        "n_posts": int(activations.shape[0]),
        "hidden_dim": int(activations.shape[1]),
        "mean_norm": float(norms.mean()),
        "median_norm": float(np.median(norms)),
        "max_norm": float(norms.max()),
        "min_norm": float(norms.min()),
    }


def extract_multilayer_activations(
    texts: list[str],
    model_name: str,
    layers: list[int],
    max_length: int = 128,
    batch_size: int = 4,
    device: str = "mps",
    dtype_name: str = "bfloat16",
    log_every_batches: int = 25,
) -> dict[int, np.ndarray]:
    """
    Mean-pooled residual-stream activations at several layers in a SINGLE forward
    pass — we hook all requested layers at once, so capturing N layers costs the
    same as capturing one. This is why the layer sweep (§3 of the handover) is
    nearly free.

    Returns
    -------
    dict mapping layer_idx -> np.ndarray of shape (len(texts), hidden_dim), float32.
    """
    device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name)

    tokenizer, model, decoder_layers = load_base_lm(model_name, device, dtype)

    for li in layers:
        if li < 0 or li >= len(decoder_layers):
            raise ValueError(f"layer {li} out of range; model has {len(decoder_layers)} layers.")
    print(f"  hooking layers {layers} of {len(decoder_layers)} in one pass")

    captured: dict[int, "torch.Tensor"] = {}

    def make_hook(layer_id: int):
        def hook(module, inputs, output):
            captured[layer_id] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [decoder_layers[li].register_forward_hook(make_hook(li)) for li in layers]

    n = len(texts)
    pooled_by_layer: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    t_start = time.time()
    try:
        for b_idx, start in enumerate(range(0, n, batch_size)):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                _ = model(**enc, use_cache=False)

            mask = enc.attention_mask.unsqueeze(-1)   # (B, T, 1)
            for li in layers:
                acts = captured[li]
                m = mask.to(acts.dtype)
                pooled = ((acts * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)).float()
                pooled_by_layer[li].append(pooled.cpu().numpy().astype(np.float32))

            done = min(start + batch_size, n)
            if (b_idx + 1) % log_every_batches == 0 or done >= n:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                eta = max(n - done, 0) / max(rate, 1e-6)
                print(f"  {done:4d}/{n} posts | {rate:5.2f} posts/s | "
                      f"elapsed {elapsed:5.0f}s | eta {eta:5.0f}s")
    finally:
        for h in handles:
            h.remove()

    del model
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return {li: np.vstack(pooled_by_layer[li]) for li in layers}

