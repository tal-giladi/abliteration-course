# Lesson 07 - Orthogonal projection
# Read lessons/module-03/lesson-01.md before filling this in.
# No model needed for this one — pure vector math on small synthetic tensors.

import torch
import torch.nn.functional as F


def ablate_vector(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of v along `direction`. Works for a single vector
    [hidden] or a batch [n, hidden]; `direction` is always [hidden] and gets
    normalized internally.
    """
    # TODO: normalize `direction` to a unit vector (F.normalize), compute the
    # coefficient v @ d (a scalar for a single vector, a [n] tensor for a
    # batch), and subtract coeff * d from v. Use v.dim() to tell the two
    # shapes apart — the batch case needs coeff.unsqueeze(-1) before
    # multiplying so it broadcasts against d correctly.
    raise NotImplementedError
