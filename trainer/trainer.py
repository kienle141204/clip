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

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    def _forward_batch(self, batch):
        images = batch["image"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        return self.model(images, input_ids, attention_mask)

    # ------------------------------------------------------------------
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
                print(f"  [epoch {epoch} step {step}/{len(self.train_loader)}] loss={avg:.4f}")

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_loss: float):
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
            print(f"  --> Best model saved (val_loss={val_loss:.4f})")

    # ------------------------------------------------------------------
    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.evaluate()
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            self.save_checkpoint(epoch, val_loss)
