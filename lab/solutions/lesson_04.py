import torch
from lib.direction import get_residuals_per_prompt


def get_residuals_mean(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    return get_residuals_per_prompt(prompts, batch_size=batch_size).mean(dim=0)
