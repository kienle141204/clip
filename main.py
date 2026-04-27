import argparse
import torch
from transformers import DistilBertTokenizer

from config import Config
from data_loader import build_loaders
from model import CLIPModel
from utils import get_train_transforms, get_val_transforms
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP on Flickr8k")
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--captions_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    return parser.parse_args()


def apply_args(config: Config, args) -> Config:
    for field, value in vars(args).items():
        if value is not None and hasattr(config, field):
            setattr(config, field, value)
    if args.lr is not None:
        config.learning_rate = args.lr
    return config


def main():
    args = parse_args()
    config = apply_args(Config(), args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config}")

    tokenizer = DistilBertTokenizer.from_pretrained(config.text_encoder)

    train_loader, val_loader, test_loader = build_loaders(
        config,
        tokenizer,
        train_transform=get_train_transforms(config.image_size),
        val_transform=get_val_transforms(config.image_size),
    )
    print(
        f"Dataset splits — train: {len(train_loader.dataset)} | "
        f"val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}"
    )

    model = CLIPModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()

    # Final evaluation on test set
    print("\n--- Test set evaluation ---")
    trainer.val_loader = test_loader
    _, test_metrics = trainer.evaluate()
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
