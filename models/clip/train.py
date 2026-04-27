import argparse

import torch
from transformers import DistilBertTokenizer

from data_loader import build_loaders
from utils import get_train_transforms, get_val_transforms
from models.clip.config import ClipConfig
from models.clip import CLIPModel
from models.clip.trainer import Trainer


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Train CLIP on Flickr8k")
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--captions_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    return parser.parse_args(argv)


def _apply_args(config: ClipConfig, args) -> ClipConfig:
    for key, value in vars(args).items():
        if value is None:
            continue
        if key == "lr":
            config.learning_rate = value
        elif hasattr(config, key):
            setattr(config, key, value)
    return config


def run(argv=None):
    args = _parse_args(argv)
    config = _apply_args(ClipConfig(), args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)
    print(f"Config : {config}", flush=True)

    tokenizer = DistilBertTokenizer.from_pretrained(config.text_encoder)

    train_loader, val_loader, test_loader = build_loaders(
        config,
        tokenizer,
        train_transform=get_train_transforms(config.image_size),
        val_transform=get_val_transforms(config.image_size),
    )
    print(
        f"Splits — train: {len(train_loader.dataset)} | "
        f"val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}",
        flush=True,
    )

    model = CLIPModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}", flush=True)

    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()

    print("\n--- Test set evaluation ---", flush=True)
    trainer.val_loader = test_loader
    _, test_metrics = trainer.evaluate()
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
