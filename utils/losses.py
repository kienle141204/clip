import torch
import torch.nn.functional as F


def multi_positive_contrastive_loss(image_emb, text_emb, scale, captions_per_image):
    """Symmetric multi-positive InfoNCE for K captions per image (SupCon-style i2t).

    Each image has K matching captions; each caption matches exactly one image.
    Captions are assumed to be ordered as
        [img0_cap0, img0_cap1, ..., img0_capK-1, img1_cap0, ...]

    Args:
        image_emb:           (B, D)   L2-normalised image embeddings (one per unique image)
        text_emb:            (B*K, D) L2-normalised text embeddings (K captions per image)
        scale:               scalar; logit_scale.exp() — multiplies cosine sims
        captions_per_image:  K
    """
    B, _ = image_emb.shape
    K = captions_per_image
    assert text_emb.size(0) == B * K, (
        f"Expected {B * K} text embeddings, got {text_emb.size(0)}"
    )

    logits = (image_emb @ text_emb.T) * scale  # (B, B*K)

    # Each text's positive image is text_idx // K
    text_to_image = torch.arange(B * K, device=logits.device) // K  # (B*K,)

    # t2i: each text has 1 positive image — standard CE
    loss_t2i = F.cross_entropy(logits.T, text_to_image)

    # i2t: each image has K positives — SupCon-style mean of -log_softmax
    log_probs = F.log_softmax(logits, dim=1)             # (B, B*K)
    pos_indices = (
        torch.arange(B, device=logits.device).unsqueeze(1) * K
        + torch.arange(K, device=logits.device).unsqueeze(0)
    )                                                     # (B, K)
    pos_log_probs = log_probs.gather(1, pos_indices)     # (B, K)
    loss_i2t = -pos_log_probs.mean()

    return (loss_i2t + loss_t2i) / 2
