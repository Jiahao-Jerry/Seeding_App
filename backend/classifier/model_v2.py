import torch
import torch.nn as nn
from transformers import AutoModel

AXIS_NAMES = [
    "reading_level", "concreteness", "narrativity", "hedging", "tone",
    "warmth", "self_disclosure", "casualness", "humor",
]

BACKBONE_V2 = "roberta-base"  # Option 5: bigger backbone than the v1 DistilBERT


class DeltaRegressor(nn.Module):
    """
    Option 1: trained directly on (original, rewrite) -> delta, instead of
    absolute per-post scoring + subtraction. Shared/siamese encoder scores
    both texts; the head sees both [CLS] embeddings plus their difference.
    """
    def __init__(self, backbone=BACKBONE_V2, n_axes=len(AXIS_NAMES), dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_axes),
        )

    def _encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]  # [CLS]

    def forward(self, orig_ids, orig_mask, new_ids, new_mask):
        orig_cls = self._encode(orig_ids, orig_mask)
        new_cls = self._encode(new_ids, new_mask)
        diff = new_cls - orig_cls
        combined = self.dropout(torch.cat([orig_cls, new_cls, diff], dim=-1))
        return torch.tanh(self.head(combined))  # bounded to [-1, 1], matches realistic delta range
