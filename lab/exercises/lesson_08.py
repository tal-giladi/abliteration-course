# Lesson 08 - From vectors to weight matrices
# Read lessons/module-03/lesson-02.md before filling this in.
# No model needed for the math — the checker loads the real model only to
# confirm your functions work on real weight shapes, no forward pass.

import torch
import torch.nn.functional as F


def ablate_matrix_output(W: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Project `direction` out of every output W can produce: W is an
    nn.Linear weight [out_features, in_features], y = W @ x, and `direction`
    lives in the [out_features] space. `weight` scales how much of the
    projection to apply (1.0 = fully remove, 0.0 = no-op).
    """
    # TODO: normalize `direction`, then return W - weight * outer(d, d @ W).
    raise NotImplementedError


def ablate_matrix_input(E: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Project `direction` out of every ROW of E, e.g. embed_tokens.weight
    [vocab_size, hidden_size], where each row IS a residual-stream vector
    directly (not the output of a y = Wx computation).
    """
    # TODO: normalize `direction`, then return E - weight * outer(E @ d, d).
    raise NotImplementedError
