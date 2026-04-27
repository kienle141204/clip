import torch
import torch.nn.functional as F


def contrastive_loss(image_emb, text_emb, temperature, image_ids=None):
    """Symmetric InfoNCE / CLIP contrastive loss with optional false-negative masking.

    Args:
        image_emb:   (N, D) L2-normalised image embeddings
        text_emb:    (N, D) L2-normalised text embeddings
        temperature: scalar, learned or fixed
        image_ids:   (N,) integer tensor — when provided, pairs sharing the same
                     image_id (different captions of the same image) are masked out
                     as false negatives instead of being pushed apart.
    """
    logits = (image_emb @ text_emb.T) * temperature  # (N, N)

    if image_ids is not None:
        # Positions where two samples share the same image but are NOT the diagonal
        # are false negatives — set them to -inf so they don't contribute to the loss.
        same_image = image_ids.unsqueeze(1) == image_ids.unsqueeze(0)  # (N, N)
        same_image.fill_diagonal_(False)
        logits = logits.masked_fill(same_image, float("-inf"))

    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
