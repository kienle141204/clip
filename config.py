from dataclasses import dataclass


@dataclass
class Config:
    # Data paths
    images_dir: str = "data/Flickr8k_Dataset/images"
    captions_file: str = "data/Flickr8k_Dataset/captions.txt"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42

    # Model
    image_encoder: str = "resnet50"
    text_encoder: str = "distilbert-base-uncased"
    embed_dim: int = 256
    max_seq_len: int = 77
    image_size: int = 224

    # Training
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    clip_grad_norm: float = 1.0
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
