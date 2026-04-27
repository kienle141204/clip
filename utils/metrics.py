from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch


def recall_at_k(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    image_ids: Optional[torch.Tensor] = None,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Recall@K for image-to-text and text-to-image retrieval.

    Supports the standard multi-caption protocol (5 captions per image):
      - i2t: for each unique image, a hit@k means at least one of its captions
             appears in the top-k ranked texts.
      - t2i: for each text, a hit@k means its matching image appears in the
             top-k ranked (unique) images.

    Args:
        image_emb: (N, D) — may contain duplicate rows when each caption of
                   the same image produces a separate sample.
        text_emb:  (N, D)
        image_ids: (N,) integer tensor identifying which unique image each
                   sample belongs to.  When None, falls back to the legacy
                   1-to-1 diagonal assumption.
    """
    if image_ids is None:
        return _recall_diagonal(image_emb, text_emb, ks)

    # --- Deduplicate image embeddings (keep first occurrence) ---
    seen: dict = {}
    unique_indices = []
    for i, id_ in enumerate(image_ids.tolist()):
        if id_ not in seen:
            seen[id_] = len(unique_indices)
            unique_indices.append(i)

    unique_image_emb = image_emb[unique_indices]          # (M, D)
    id_to_uniq = seen                                     # original_id -> row in unique_image_emb

    # For each text sample, index of its matching unique image row
    text_to_uniq_image = torch.tensor(
        [id_to_uniq[id_.item()] for id_ in image_ids],
        device=image_emb.device,
    )  # (N,)

    # For each unique image row, list of matching text indices
    image_to_texts: Dict[int, list] = defaultdict(list)
    for j, id_ in enumerate(image_ids.tolist()):
        image_to_texts[id_to_uniq[id_]].append(j)

    # Similarity: unique images (M) vs all texts (N)
    sim = unique_image_emb @ text_emb.T  # (M, N)
    M, N = sim.shape

    results = {}
    for k in ks:
        # --- t2i: for each text, rank of its matching unique image ---
        k_t2i = min(k, M)
        topk_t2i = sim.T.topk(k_t2i, dim=1).indices          # (N, k_t2i)
        hits_t2i = (topk_t2i == text_to_uniq_image.unsqueeze(1)).any(1).float().mean().item()
        results[f"t2i_R@{k}"] = hits_t2i

        # --- i2t: for each unique image, any of its captions in top-k texts ---
        k_i2t = min(k, N)
        topk_i2t = sim.topk(k_i2t, dim=1).indices             # (M, k_i2t)
        hits_i2t = []
        for i in range(M):
            correct = set(image_to_texts[i])
            retrieved = set(topk_i2t[i].tolist())
            hits_i2t.append(bool(correct & retrieved))
        results[f"i2t_R@{k}"] = sum(hits_i2t) / M

    return results


def _recall_diagonal(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    ks: Tuple[int, ...],
) -> Dict[str, float]:
    """Legacy 1-to-1 recall (image_emb[i] matches text_emb[i])."""
    sim = image_emb @ text_emb.T
    n = sim.size(0)
    gt = torch.arange(n, device=sim.device)

    results = {}
    for k in ks:
        k = min(k, n)
        topk_i2t = sim.topk(k, dim=1).indices
        results[f"i2t_R@{k}"] = (topk_i2t == gt.unsqueeze(1)).any(1).float().mean().item()

        topk_t2i = sim.T.topk(k, dim=1).indices
        results[f"t2i_R@{k}"] = (topk_t2i == gt.unsqueeze(1)).any(1).float().mean().item()

    return results
