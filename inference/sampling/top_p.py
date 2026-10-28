"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    # Keep the token that crosses the threshold; remove only tokens after it.
    remove_mask = (cumulative_probs - probs) >= p

    remove_mask = torch.zeros_like(remove_mask).scatter(-1, sorted_indices, remove_mask)
    filtered_logits = logits.clone()
    filtered_logits[remove_mask] = float("-inf")
    return filtered_logits
