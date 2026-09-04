# Lesson 05 - Computing the direction
# Read lessons/module-02/lesson-02.md before filling this in.

import torch
import torch.nn.functional as F


def compute_direction(mean_a: torch.Tensor, mean_b: torch.Tensor) -> torch.Tensor:
    """The (normalized) direction that separates group A from group B.

    mean_a, mean_b: [num_layers + 1, hidden_size] — e.g. get_residuals_mean()
    over a harmful prompt set and a harmless one. Returns a unit vector per
    layer, same shape as the inputs.
    """
    # TODO: subtract mean_b from mean_a, then normalize the result to unit
    # length along the last dimension, independently at every layer.
    raise NotImplementedError
