import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b / (a.norm() * b.norm())).item()


def mean_difference(group_a: torch.Tensor, group_b: torch.Tensor) -> torch.Tensor:
    return group_a.mean(dim=0) - group_b.mean(dim=0)
