import torch
import torch.nn.functional as F


def contrastive_loss(image_emb, text_emb, temperature):
    """Symmetric InfoNCE / CLIP contrastive loss.

    Args:
        image_emb: (N, D) L2-normalised image embeddings
        text_emb:  (N, D) L2-normalised text embeddings
        temperature: scalar, learned or fixed
    """
    logits = (image_emb @ text_emb.T) * temperature  # (N, N)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
