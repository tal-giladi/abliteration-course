# Lesson 06 - Sanity-checking a direction
# Read lessons/module-02/lesson-03.md before filling this in.

import torch
import torch.nn.functional as F


def projection_scores(residuals: torch.Tensor, direction: torch.Tensor, layer: int) -> torch.Tensor:
    """How far along `direction` each prompt's residual sits, at one layer.

    residuals: [num_prompts, num_layers + 1, hidden_size] (per-prompt, not mean)
    direction: [num_layers + 1, hidden_size]
    Returns a Tensor [num_prompts] of raw dot products (not normalized —
    the sign and relative magnitude are what matter for separation).
    """
    # TODO: pull out this layer's direction vector, re-normalize it to unit
    # length (safe even if it already is), and dot it against every
    # prompt's residual at this layer in one batched matrix-vector product.
    raise NotImplementedError
