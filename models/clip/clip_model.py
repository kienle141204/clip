import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

# Prevent logit scale from growing unboundedly (equivalent to temperature < 0.01)
_MAX_LOGIT_SCALE = math.log(100.0)


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(config.image_encoder, config.embed_dim)
        self.text_encoder = TextEncoder(config.text_encoder, config.embed_dim)
        # logit_scale = log(1/temperature) so that similarity * exp(logit_scale)
        # gives sharp distributions. Initialised to log(1/0.07) ≈ 2.66.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / config.temperature)))

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=_MAX_LOGIT_SCALE)
        return image_emb, text_emb, scale
