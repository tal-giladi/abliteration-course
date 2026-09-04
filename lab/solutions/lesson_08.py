import torch
import torch.nn.functional as F


def ablate_matrix_output(W: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    d = F.normalize(direction, dim=-1)
    return W - weight * torch.outer(d, d @ W)


def ablate_matrix_input(E: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    d = F.normalize(direction, dim=-1)
    return E - weight * torch.outer(E @ d, d)
