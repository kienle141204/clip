import torch.nn as nn
import torchvision.models as models


_BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
}


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet50", embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        if model_name not in _BACKBONES:
            raise ValueError(f"Unsupported image encoder: {model_name}")

        factory, weights, feat_dim = _BACKBONES[model_name]
        backbone = factory(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.projection(features)
