import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", embed_dim: int = 256):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls)
