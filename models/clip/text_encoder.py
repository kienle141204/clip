import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Masked mean pooling: average over non-padding tokens
        hidden = outputs.last_hidden_state          # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float() # (B, L, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, H)
        return self.projection(pooled)
