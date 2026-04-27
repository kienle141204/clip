import torch
from typing import Dict, Tuple


def recall_at_k(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute Recall@K for image-to-text and text-to-image retrieval.

    Assumes a 1-to-1 correspondence: image_emb[i] matches text_emb[i].
    """
    sim = image_emb @ text_emb.T  # (N, N)
    n = sim.size(0)
    gt = torch.arange(n, device=sim.device)

    results = {}
    for k in ks:
        k = min(k, n)
        # image -> text
        topk_i2t = sim.topk(k, dim=1).indices
        r_i2t = (topk_i2t == gt.unsqueeze(1)).any(1).float().mean().item()
        results[f"i2t_R@{k}"] = r_i2t

        # text -> image
        topk_t2i = sim.T.topk(k, dim=1).indices
        r_t2i = (topk_t2i == gt.unsqueeze(1)).any(1).float().mean().item()
        results[f"t2i_R@{k}"] = r_t2i

    return results
