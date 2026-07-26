import torch
import torch.nn as nn
from transformers import AutoModel

AXIS_NAMES = [
    "reading_level", "concreteness", "narrativity", "hedging", "tone",
    "warmth", "self_disclosure", "casualness", "humor",
]

BACKBONE = "distilbert-base-uncased"


class AxisRegressor(nn.Module):
    def __init__(self, backbone=BACKBONE, n_axes=len(AXIS_NAMES), dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_axes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] token
        cls = self.dropout(cls)
        return torch.sigmoid(self.head(cls))
