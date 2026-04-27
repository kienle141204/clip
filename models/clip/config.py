from dataclasses import dataclass


@dataclass
class ClipConfig:
    # Data paths
    images_dir: str = "data/Flickr8k_Dataset/images"
    captions_file: str = "data/Flickr8k_Dataset/captions.txt"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42

    # Model
    image_encoder: str = "resnet50"
    text_encoder: str = "distilbert-base-uncased"
    embed_dim: int = 512
    max_seq_len: int = 77
    image_size: int = 224
    captions_per_image: int = 5      # K positives per image for multi-positive InfoNCE

    # Model regularization
    dropout: float = 0.1

    # Training
    # NB: each batch step processes batch_size images × captions_per_image captions,
    # so text-side memory scales with K. Lower batch_size if OOM.
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    temperature: float = 0.07
    clip_grad_norm: float = 1.0
    num_workers: int = 4
    warmup_epochs: int = 3            # linear LR warmup for first N epochs
    freeze_backbone_epochs: int = 3   # freeze backbone for first N epochs
    early_stopping_patience: int = 8  # stop if val_loss doesn't improve for N epochs

    # Paths
    checkpoint_dir: str = "checkpoints/clip"
    log_interval: int = 50
