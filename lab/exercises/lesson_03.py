# Lesson 03 - Behavior lives in a direction, not a neuron
# Read lessons/module-01/lesson-03.md before filling this in.
# No model needed for this one — pure vector math on small synthetic tensors.

import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """The cosine of the angle between a and b: 1.0 = same direction,
    0.0 = perpendicular, -1.0 = opposite. Independent of magnitude."""
    # TODO
    raise NotImplementedError


def mean_difference(group_a: torch.Tensor, group_b: torch.Tensor) -> torch.Tensor:
    """group_a, group_b: [n, dim] batches of vectors.
    Returns mean(group_a) - mean(group_b), a single [dim] vector."""
    # TODO
    raise NotImplementedError
