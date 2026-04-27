import csv
import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        split: str = "train",
        tokenizer=None,
        transform=None,
        max_seq_len: int = 77,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len

        pairs = self._parse_captions(captions_file)
        self.data = self._split(pairs, split, train_ratio, val_ratio, seed)

    def _parse_captions(self, captions_file: str):
        """Parse captions.txt (CSV: image,caption) -> list of (image_name, caption)."""
        pairs = []
        with open(captions_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["image"].strip(), row["caption"].strip()))
        return pairs

    def _split(self, pairs, split, train_ratio, val_ratio, seed):
        # Group by image to avoid same image appearing in multiple splits
        image_to_captions = defaultdict(list)
        for image_name, caption in pairs:
            image_to_captions[image_name].append(caption)

        images = sorted(image_to_captions.keys())
        rng = random.Random(seed)
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            selected = images[:n_train]
        elif split == "val":
            selected = images[n_train : n_train + n_val]
        else:
            selected = images[n_train + n_val :]

        data = []
        for img in selected:
            for cap in image_to_captions[img]:
                data.append({"image": img, "caption": cap})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.images_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer(
            item["caption"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "caption": item["caption"],
        }


def build_loaders(config, tokenizer, train_transform, val_transform):
    common = dict(
        images_dir=config.images_dir,
        captions_file=config.captions_file,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_ds = Flickr8kDataset(**common, split="train", transform=train_transform)
    val_ds = Flickr8kDataset(**common, split="val", transform=val_transform)
    test_ds = Flickr8kDataset(**common, split="test", transform=val_transform)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
