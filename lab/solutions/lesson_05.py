import torch
import torch.nn.functional as F


def compute_direction(mean_a: torch.Tensor, mean_b: torch.Tensor) -> torch.Tensor:
    diff = mean_a - mean_b
    return F.normalize(diff, dim=-1)
