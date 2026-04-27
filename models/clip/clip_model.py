import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(config.image_encoder, config.embed_dim)
        self.text_encoder = TextEncoder(config.text_encoder, config.embed_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(config.temperature)))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        return image_emb, text_emb, self.temperature
