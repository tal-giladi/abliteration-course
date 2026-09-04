# SPDX-License-Identifier: MIT
"""The projection math, and the two places it gets applied: a single vector,
and a weight matrix that writes into the residual stream.

Everything here is the same operation — "remove the component along d" —
at three different scales:
  - a vector:  v'  = v  - (v . d) d
  - a matrix:  W'  = W  - d (d^T W)      (W writes hidden_size-shaped outputs)
  - a matrix:  E'  = E  - (E d) d^T      (E's *rows* are hidden_size vectors)
The matrix forms are just the vector form applied to every row/column at
once — no new math, only new shapes. See Lesson 07 and Lesson 08.
"""

import torch
import torch.nn.functional as F


def ablate_vector(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of v along `direction`. Works for a single vector
    [hidden] or a batch [n, hidden]; `direction` is always [hidden]."""
    d = F.normalize(direction, dim=-1)
    coeff = v @ d  # scalar, or [n]
    return v - coeff.unsqueeze(-1) * d if v.dim() > 1 else v - coeff * d


def ablate_matrix_output(W: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Project `direction` out of a matrix whose OUTPUT is a residual-stream
    vector, e.g. self_attn.o_proj.weight or mlp.down_proj.weight — both are
    nn.Linear weights of shape [hidden_size, in_features], where y = W @ x.

    `weight` is how much of the projection to apply: 1.0 = fully remove the
    direction from every possible output, 0.0 = no change. Heretic tunes this
    per layer instead of using a flat 1.0 everywhere (Lesson 12).
    """
    d = F.normalize(direction, dim=-1)
    return W - weight * torch.outer(d, d @ W)


def ablate_matrix_input(E: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Project `direction` out of a matrix whose ROWS are residual-stream
    vectors, e.g. embed_tokens.weight, shape [vocab_size, hidden_size]."""
    d = F.normalize(direction, dim=-1)
    return E - weight * torch.outer(E @ d, d)


def lora_rank1_factors(W: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Express ablate_matrix_output(W, direction, weight=1.0) - W as a rank-1
    product B @ A, instead of editing W directly.

    Returns (A, B) with A: [1, in_features], B: [out_features, 1], such that
    `W + B @ A` equals `ablate_matrix_output(W, direction)`. This is the
    entire reason Heretic can "try" an ablation without touching the base
    model's weights: it loads this pair as a rank-1 LoRA adapter instead
    (Lesson 09).
    """
    d = F.normalize(direction, dim=-1)
    A = (d @ W).unsqueeze(0)
    B = (-d).unsqueeze(1)
    return A, B


def ablation_forward_hook(direction: torch.Tensor):
    """A forward hook that ablates a decoder layer's *output* live, without
    touching any weights. Register on `model.model.layers[i]`. Used for
    Lesson 10's live-hook version before Lesson 11 makes the edit permanent."""
    d = F.normalize(direction, dim=-1)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        ablated = ablate_vector(hidden, d)
        return (ablated, *output[1:]) if isinstance(output, tuple) else ablated

    return hook


def merge_into_model(model, directions: torch.Tensor, weights: list[float] | None = None) -> None:
    """Permanently ablate `direction[i]` into layer i's weights, for every
    layer, in place. directions: [num_layers + 1, hidden_size] as produced by
    direction.compute_direction(). weights: one multiplier per entry in
    `directions` (default: 1.0 everywhere — see Lesson 12 for why you'd vary
    it). A weight of 0 skips that layer entirely.
    """
    layers = model.model.layers
    weights = weights or [1.0] * (len(layers) + 1)
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        if weights[0] != 0:
            emb.copy_(ablate_matrix_input(emb, directions[0], weights[0]))
        for i, layer in enumerate(layers):
            w = weights[i + 1]
            if w == 0:
                continue
            d = directions[i + 1]
            o_proj = layer.self_attn.o_proj.weight
            o_proj.copy_(ablate_matrix_output(o_proj, d, w))
            down_proj = layer.mlp.down_proj.weight
            down_proj.copy_(ablate_matrix_output(down_proj, d, w))
