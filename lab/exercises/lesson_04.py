# Lesson 04 - Harmful vs. harmless prompt sets
# Read lessons/module-02/lesson-01.md before filling this in.

import torch
from lib.direction import get_residuals_per_prompt


def get_residuals_mean(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    """Mean last-token residual stream across a set of prompts.
    Returns a Tensor [num_layers + 1, hidden_size].
    """
    # TODO: get the per-prompt residuals (reuse get_residuals_per_prompt,
    # already correct in lib.direction — this lesson isn't about
    # re-deriving that part) and collapse the prompt dimension with a mean.
    raise NotImplementedError
