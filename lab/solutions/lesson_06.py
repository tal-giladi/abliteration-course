import torch
import torch.nn.functional as F


def projection_scores(residuals: torch.Tensor, direction: torch.Tensor, layer: int) -> torch.Tensor:
    d = F.normalize(direction[layer], dim=-1)
    return residuals[:, layer, :] @ d
