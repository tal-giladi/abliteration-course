import torch
import torch.nn.functional as F


def ablate_vector(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    d = F.normalize(direction, dim=-1)
    coeff = v @ d
    return v - coeff.unsqueeze(-1) * d if v.dim() > 1 else v - coeff * d
