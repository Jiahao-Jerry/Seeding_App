"""
Fine-tune a small encoder (DistilBERT) as a 9-axis regressor on the
1,997 posts with source=="labeled" (real ground truth, not SAE/Ridge estimates).

Replaces the Qwen -> SAE -> Ridge pipeline for axis-delta verification:
directly supervised on the actual axis labels instead of hoping axis-relevant
directions fall out of an unsupervised sparse autoencoder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from backend.classifier.model import AXIS_NAMES, BACKBONE, AxisRegressor

DEVICE = "cpu"  # MPS hangs on this transformers/torch combo; DistilBERT is small enough for CPU to be fine
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 15
LR = 2e-5
VAL_FRAC = 0.15
SEED = 42
OUT_DIR = APP / "data/classifier"


class AxisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def load_data():
    df = pd.read_parquet(APP / "data/annotated_posts.parquet")
    df["source"] = df["axes_json"].apply(lambda s: json.loads(s)["reading_level"]["source"])
    labeled = df[df["source"] == "labeled"].reset_index(drop=True)

    texts = labeled["text"].tolist()
    axes = json_col = labeled["axes_json"].apply(json.loads)
    y = np.zeros((len(labeled), len(AXIS_NAMES)), dtype=np.float32)
    for i, ax in enumerate(AXIS_NAMES):
        y[:, i] = axes.apply(lambda d: d[ax]["score"]).values
    return texts, y


def split(texts, y, val_frac=VAL_FRAC, seed=SEED):
    n = len(texts)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (
        [texts[i] for i in train_idx], y[train_idx],
        [texts[i] for i in val_idx], y[val_idx],
    )


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            out = model(input_ids, attn).cpu().numpy()
            preds.append(out)
            targets.append(batch["labels"].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse_per_axis = ((preds - targets) ** 2).mean(axis=0)
    corr_per_axis = []
    for i in range(len(AXIS_NAMES)):
        r, _ = pearsonr(preds[:, i], targets[:, i])
        corr_per_axis.append(r)
    return mse_per_axis, corr_per_axis, preds, targets


def main():
    print(f"Device: {DEVICE}")
    texts, y = load_data()
    print(f"Loaded {len(texts)} labeled posts")
    train_texts, train_y, val_texts, val_y = split(texts, y)
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    train_ds = AxisDataset(train_texts, train_y, tokenizer)
    val_ds = AxisDataset(val_texts, val_y, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AxisRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(input_ids, attn)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_ds)
        mse_per_axis, corr_per_axis, _, _ = evaluate(model, val_loader)
        val_mse = mse_per_axis.mean()
        print(f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  val_mse={val_mse:.4f}  "
              f"val_corr_mean={np.mean(corr_per_axis):.3f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_mse_per_axis = mse_per_axis.copy()
            best_corr_per_axis = list(corr_per_axis)

    print(f"\nBest epoch: {best_epoch}  val_mse={best_val_mse:.4f}")
    print(f"{'axis':18s} {'MSE':>8} {'Pearson r':>10}")
    for i, ax in enumerate(AXIS_NAMES):
        print(f"{ax:18s} {best_mse_per_axis[i]:8.4f} {best_corr_per_axis[i]:10.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / "axis_regressor.pt")
    tokenizer.save_pretrained(OUT_DIR / "tokenizer")
    metrics = {
        "backbone": BACKBONE,
        "best_epoch": best_epoch,
        "val_mse_mean": float(best_val_mse),
        "per_axis": {
            ax: {"mse": float(best_mse_per_axis[i]), "pearson_r": float(best_corr_per_axis[i])}
            for i, ax in enumerate(AXIS_NAMES)
        },
        "n_train": len(train_texts),
        "n_val": len(val_texts),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> {OUT_DIR / 'axis_regressor.pt'}")
    print(f"Saved metrics -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
