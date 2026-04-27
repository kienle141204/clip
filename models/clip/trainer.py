import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.losses import contrastive_loss
from utils.metrics import recall_at_k


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Pretrained backbones use a lower LR to preserve learned features
        backbone_lr = config.learning_rate * 0.1
        self.optimizer = AdamW(
            [
                {"params": model.image_encoder.backbone.parameters(), "lr": backbone_lr},
                {"params": model.image_encoder.projection.parameters()},
                {"params": model.text_encoder.encoder.parameters(), "lr": backbone_lr},
                {"params": model.text_encoder.projection.parameters()},
                {"params": [model.logit_scale]},
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self._early_stop_counter = 0

        # Freeze backbones for the first N epochs to stabilise projection heads first
        if config.freeze_backbone_epochs > 0:
            self._set_backbone_grad(requires_grad=False)
            print(f"Backbones frozen for first {config.freeze_backbone_epochs} epoch(s).", flush=True)

    def _set_backbone_grad(self, requires_grad: bool):
        for p in self.model.image_encoder.backbone.parameters():
            p.requires_grad = requires_grad
        for p in self.model.text_encoder.encoder.parameters():
            p.requires_grad = requires_grad

    def _forward_batch(self, batch):
        images = batch["image"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        return self.model(images, input_ids, attention_mask)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            image_emb, text_emb, temp = self._forward_batch(batch)
            loss = contrastive_loss(image_emb, text_emb, temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
            self.optimizer.step()
            total_loss += loss.item()

            if step % self.config.log_interval == 0:
                avg = total_loss / step
                print(f"  [epoch {epoch} step {step}/{len(self.train_loader)}] loss={avg:.4f}", flush=True)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_image_emb, all_text_emb = [], []

        for batch in self.val_loader:
            image_emb, text_emb, temp = self._forward_batch(batch)
            total_loss += contrastive_loss(image_emb, text_emb, temp).item()
            all_image_emb.append(image_emb.cpu())
            all_text_emb.append(text_emb.cpu())

        image_emb = torch.cat(all_image_emb)
        text_emb = torch.cat(all_text_emb)
        metrics = recall_at_k(image_emb, text_emb)
        return total_loss / len(self.val_loader), metrics

    def save_checkpoint(self, epoch: int, val_loss: float) -> bool:
        """Save checkpoint. Returns True if this is the new best model."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        path = os.path.join(self.config.checkpoint_dir, f"clip_epoch{epoch:03d}.pt")
        torch.save(state, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_path)
            print(f"  --> Best model saved (val_loss={val_loss:.4f})", flush=True)
            return True
        return False

    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):

            # Unfreeze backbone after warmup period
            if epoch == self.config.freeze_backbone_epochs + 1:
                self._set_backbone_grad(requires_grad=True)
                print("Backbones unfrozen.", flush=True)

            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.evaluate()
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}",
                flush=True,
            )
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}", flush=True)

            improved = self.save_checkpoint(epoch, val_loss)

            # Early stopping
            if improved:
                self._early_stop_counter = 0
            else:
                self._early_stop_counter += 1
                print(
                    f"  [early stopping] no improvement for {self._early_stop_counter}"
                    f"/{self.config.early_stopping_patience} epoch(s).",
                    flush=True,
                )
                if self._early_stop_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}.", flush=True)
                    break
