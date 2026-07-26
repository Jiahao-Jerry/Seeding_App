"""
Combined-experiment training: Option 1 (delta-based training) + Option 2
(expanded pair data, aggregated from every batch generated this session) +
Option 5 (bigger backbone, roberta-base). Separate from train.py / model.py
-- the current v1 classifier is untouched.

Data: aggregates every (original, rewrite, orig_scores, new_scores) batch
generated this session (450 rows across 5 files), MINUS any post_id that
appears in the balanced 90-pair set -- that set is held out entirely, never
trained on, so it stays a clean, apples-to-apples comparison against the v1
classifier's already-known performance on it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from backend.classifier.model_v2 import AXIS_NAMES, BACKBONE_V2, DeltaRegressor

DEVICE = "cpu"  # MPS hangs on this transformers/torch combo (established earlier this session)
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5
VAL_FRAC = 0.15
SEED = 42
OUT_DIR = APP / "data/classifier_v2"

TRAIN_FILES = [
    "data/sae2/batch100_new_prompt_results.json",
    "data/sae2/batch100_old_prompt_results.json",
    "data/sae2/batch50_old_prompt_v2_results.json",
    "data/sae2/narrativity_ab_new_results.json",
    "data/sae2/narrativity_ab_old_results.json",
]
HELDOUT_EVAL_FILE = "data/sae2/balanced_90_pairs_topped_up.json"


class PairDataset(Dataset):
    def __init__(self, orig_texts, new_texts, deltas, tokenizer):
        self.orig_texts = orig_texts
        self.new_texts = new_texts
        self.deltas = deltas
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.orig_texts)

    def __getitem__(self, idx):
        orig_enc = self.tokenizer(self.orig_texts[idx], truncation=True, max_length=MAX_LEN,
                                   padding="max_length", return_tensors="pt")
        new_enc = self.tokenizer(self.new_texts[idx], truncation=True, max_length=MAX_LEN,
                                  padding="max_length", return_tensors="pt")
        return {
            "orig_ids": orig_enc["input_ids"].squeeze(0), "orig_mask": orig_enc["attention_mask"].squeeze(0),
            "new_ids": new_enc["input_ids"].squeeze(0), "new_mask": new_enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.deltas[idx], dtype=torch.float32),
        }


def load_training_pairs():
    heldout_pids = {str(r["post_id"]) for r in json.loads((APP / HELDOUT_EVAL_FILE).read_text())}

    orig_texts, new_texts, deltas = [], [], []
    post_ids = []
    n_dropped = 0
    for f in TRAIN_FILES:
        rows = json.loads((APP / f).read_text())
        for r in rows:
            if str(r["post_id"]) in heldout_pids:
                n_dropped += 1
                continue
            orig_texts.append(r["original_text"])
            new_texts.append(r["rewritten_text"])
            delta = np.array([r["new_scores"][ax] - r["orig_scores"][ax] for ax in AXIS_NAMES], dtype=np.float32)
            deltas.append(delta)
            post_ids.append(str(r["post_id"]))

    print(f"Loaded {len(orig_texts)} training pairs ({n_dropped} dropped for held-out-set leakage)")
    return orig_texts, new_texts, np.array(deltas), post_ids


def split_by_post_id(orig_texts, new_texts, deltas, post_ids, val_frac=VAL_FRAC, seed=SEED):
    unique_pids = sorted(set(post_ids))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_pids)
    n_val_pids = max(1, int(len(unique_pids) * val_frac))
    val_pid_set = set(shuffled[:n_val_pids])

    train_idx = [i for i, p in enumerate(post_ids) if p not in val_pid_set]
    val_idx = [i for i, p in enumerate(post_ids) if p in val_pid_set]

    return (
        [orig_texts[i] for i in train_idx], [new_texts[i] for i in train_idx], deltas[train_idx],
        [orig_texts[i] for i in val_idx], [new_texts[i] for i in val_idx], deltas[val_idx],
    )


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["orig_ids"].to(DEVICE), batch["orig_mask"].to(DEVICE),
                batch["new_ids"].to(DEVICE), batch["new_mask"].to(DEVICE),
            ).cpu().numpy()
            preds.append(out)
            targets.append(batch["labels"].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse_per_axis = ((preds - targets) ** 2).mean(axis=0)
    corr_per_axis = []
    for i in range(len(AXIS_NAMES)):
        if np.std(preds[:, i]) < 1e-8 or np.std(targets[:, i]) < 1e-8:
            corr_per_axis.append(0.0)
            continue
        r, _ = pearsonr(preds[:, i], targets[:, i])
        corr_per_axis.append(r if not np.isnan(r) else 0.0)
    return mse_per_axis, corr_per_axis


def main():
    print(f"Device: {DEVICE}  Backbone: {BACKBONE_V2}")
    orig_texts, new_texts, deltas, post_ids = load_training_pairs()
    train_o, train_n, train_d, val_o, val_n, val_d = split_by_post_id(orig_texts, new_texts, deltas, post_ids)
    print(f"Train: {len(train_o)}  Val: {len(val_o)} (split by post_id, no post appears in both)")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_V2)
    train_ds = PairDataset(train_o, train_n, train_d, tokenizer)
    val_ds = PairDataset(val_o, val_n, val_d, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeltaRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    best_mse_per_axis = None
    best_corr_per_axis = None
    start_epoch = 1

    ckpt_file = OUT_DIR / "train_checkpoint.pt"
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["next_epoch"]
        best_val_mse = ckpt["best_val_mse"]
        best_state = ckpt["best_state"]
        best_epoch = ckpt["best_epoch"]
        best_mse_per_axis = ckpt["best_mse_per_axis"]
        best_corr_per_axis = ckpt["best_corr_per_axis"]
        print(f"Resuming from checkpoint: starting at epoch {start_epoch}, "
              f"best so far epoch {best_epoch} (val_mse={best_val_mse:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(
                batch["orig_ids"].to(DEVICE), batch["orig_mask"].to(DEVICE),
                batch["new_ids"].to(DEVICE), batch["new_mask"].to(DEVICE),
            )
            labels = batch["labels"].to(DEVICE)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_ds)
        mse_per_axis, corr_per_axis = evaluate(model, val_loader)
        val_mse = mse_per_axis.mean()
        print(f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  val_mse={val_mse:.4f}  "
              f"val_corr_mean={np.mean(corr_per_axis):.3f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_mse_per_axis = mse_per_axis.copy()
            best_corr_per_axis = list(corr_per_axis)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "next_epoch": epoch + 1, "best_val_mse": best_val_mse, "best_state": best_state,
            "best_epoch": best_epoch, "best_mse_per_axis": best_mse_per_axis, "best_corr_per_axis": best_corr_per_axis,
        }, ckpt_file)
        print(f"  (checkpoint saved: epoch {epoch} done)")

    print(f"\nBest epoch: {best_epoch}  val_mse={best_val_mse:.4f}")
    print(f"{'axis':18s} {'MSE':>8} {'Pearson r':>10}")
    for i, ax in enumerate(AXIS_NAMES):
        print(f"{ax:18s} {best_mse_per_axis[i]:8.4f} {best_corr_per_axis[i]:10.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / "delta_regressor.pt")
    tokenizer.save_pretrained(OUT_DIR / "tokenizer")
    metrics = {
        "backbone": BACKBONE_V2,
        "best_epoch": best_epoch,
        "val_mse_mean": float(best_val_mse),
        "per_axis": {
            ax: {"mse": float(best_mse_per_axis[i]), "pearson_r": float(best_corr_per_axis[i])}
            for i, ax in enumerate(AXIS_NAMES)
        },
        "n_train": len(train_o), "n_val": len(val_o),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> {OUT_DIR / 'delta_regressor.pt'}")
    print(f"Saved metrics -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
