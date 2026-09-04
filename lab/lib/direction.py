# SPDX-License-Identifier: MIT
"""Extracting a behavior direction from a model's hidden states.

A "direction" in this course is always a Tensor of shape [num_layers + 1,
hidden_size]: one vector per residual-stream checkpoint — index 0 is the
embedding output (before layer 0), index i is the output of decoder layer
i - 1. That is exactly the shape `output_hidden_states=True` gives you, which
is deliberate: you never have to reindex between "a hidden state" and "a
direction".
"""

import torch
import torch.nn.functional as F

from .common import DEVICE, encode_chat, load_model


@torch.no_grad()
def get_residuals_per_prompt(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    """Last-token residual stream for every prompt, at every layer.

    Returns a Tensor [len(prompts), num_layers + 1, hidden_size].
    Left-padding (set up in common.encode_chat) is what makes `[:, -1, :]`
    safe: the last column is always the last real token, never a pad token,
    no matter how short a prompt in the batch is.
    """
    model = load_model()
    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = encode_chat(batch)
        out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple(num_layers + 1) of [batch, seq, hidden]
        stacked = torch.stack(out.hidden_states, dim=1)  # [batch, layers+1, seq, hidden]
        rows.append(stacked[:, :, -1, :])  # last token (safe: left-padded)
    return torch.cat(rows, dim=0).to(DEVICE)


def get_residuals_mean(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    """Mean last-token residual stream across a set of prompts.

    Returns a Tensor [num_layers + 1, hidden_size].
    """
    return get_residuals_per_prompt(prompts, batch_size=batch_size).mean(dim=0)


def compute_direction(mean_a: torch.Tensor, mean_b: torch.Tensor) -> torch.Tensor:
    """The (normalized) direction that separates group A from group B.

    mean_a, mean_b: [num_layers + 1, hidden_size] — e.g. get_residuals_mean()
    over a harmful prompt set and a harmless one. Returns a unit vector per
    layer, same shape as the inputs.
    """
    diff = mean_a - mean_b
    return F.normalize(diff, dim=-1)


def projection_scores(residuals: torch.Tensor, direction: torch.Tensor, layer: int) -> torch.Tensor:
    """How far along `direction` each prompt's residual sits, at one layer.

    residuals: [num_prompts, num_layers + 1, hidden_size] (per-prompt, not mean)
    direction: [num_layers + 1, hidden_size]
    Returns a Tensor [num_prompts] of raw dot products (not normalized —
    the sign and relative magnitude are what matter for separation).
    """
    d = F.normalize(direction[layer], dim=-1)
    return residuals[:, layer, :] @ d
