"""
SAE2 Qwen extraction driver: run the multilayer extractor over the curated
corpus and cache one (9500, 3584) matrix per candidate layer.

All candidate layers come out of a SINGLE forward pass (see
activations.extract_multilayer_activations), so this is ~one extraction's worth
of time (~45 min on MPS for 9,500 posts), not one-per-layer. Re-runs reuse the
cached files and skip extraction.

Run:  python backend/sae/extract_qwen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))


def main() -> None:
    from config.settings import (
        SAE2_QWEN_MODEL, SAE2_QWEN_LAYERS, SAE2_QWEN_MAX_TOKENS,
        SAE2_QWEN_BATCH, SAE2_QWEN_DEVICE, SAE2_QWEN_DTYPE, SAE2_QWEN_DIR,
    )
    from backend.sae.dataset import load_dataset
    from backend.sae.activations import extract_multilayer_activations, activations_summary

    df, _bge = load_dataset()
    out_dir = (APP_ROOT / SAE2_QWEN_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [li for li in SAE2_QWEN_LAYERS if not (out_dir / f"qwen_L{li}.npy").exists()]
    if not todo:
        print(f"all layers {SAE2_QWEN_LAYERS} already cached in {out_dir}")
        return

    print(f"extracting layers {todo} from {SAE2_QWEN_MODEL} over {len(df)} posts…")
    acts = extract_multilayer_activations(
        texts=df["text"].astype(str).tolist(),
        model_name=SAE2_QWEN_MODEL,
        layers=todo,
        max_length=SAE2_QWEN_MAX_TOKENS,
        batch_size=SAE2_QWEN_BATCH,
        device=SAE2_QWEN_DEVICE,
        dtype_name=SAE2_QWEN_DTYPE,
    )
    for li, mat in acts.items():
        np.save(out_dir / f"qwen_L{li}.npy", mat)
        print(f"  layer {li}: {activations_summary(mat)}  → qwen_L{li}.npy")


if __name__ == "__main__":
    main()
