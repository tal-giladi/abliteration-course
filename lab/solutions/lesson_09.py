import torch
import torch.nn.functional as F


def lora_rank1_factors(W: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = F.normalize(direction, dim=-1)
    A = (d @ W).unsqueeze(0)
    B = (-d).unsqueeze(1)
    return A, B
