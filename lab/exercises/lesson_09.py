# Lesson 09 - Why a rank-1 LoRA instead of a weight edit
# Read lessons/module-03/lesson-03.md before filling this in.

import torch
import torch.nn.functional as F


def lora_rank1_factors(W: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor ablate_matrix_output(W, direction, weight=1.0) - W as a rank-1
    product B @ A, instead of editing W directly.

    Returns (A, B) with A: [1, in_features], B: [out_features, 1], such that
    `W + B @ A` equals `ablate_matrix_output(W, direction, weight=1.0)`.
    """
    # TODO: normalize `direction`, then set
    #   A = (d @ W).unsqueeze(0)   # [1, in_features]
    #   B = (-d).unsqueeze(1)      # [out_features, 1]
    raise NotImplementedError
