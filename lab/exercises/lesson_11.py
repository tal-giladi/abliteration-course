# Lesson 11 - Making the edit permanent
# Read lessons/module-04/lesson-02.md before filling this in.

import torch
from lib.ablate import ablate_matrix_input, ablate_matrix_output


def merge_into_model(model, directions: torch.Tensor, weights: list[float] | None = None) -> None:
    """Permanently ablate directions[i] into checkpoint i's weights, for
    every checkpoint, in place. No return value — `model` is mutated.

    directions: [num_layers + 1, hidden_size], e.g. from
    lib.direction.compute_direction(). weights: one multiplier per entry in
    `directions` (default: 1.0 everywhere — Lesson 12 is what makes this
    list interesting). A weight of exactly 0 skips that checkpoint.
    """
    # TODO:
    #  - default `weights` to [1.0] * (num_layers + 1) if None
    #  - inside torch.no_grad():
    #      - if weights[0] != 0: ablate model.model.embed_tokens.weight with
    #        ablate_matrix_input(emb, directions[0], weights[0]), and write
    #        it back with Tensor.copy_() (NOT plain reassignment — that
    #        would rebind a local variable, not edit the model's parameter)
    #      - for each decoder layer i: if weights[i + 1] == 0, skip it;
    #        otherwise ablate self_attn.o_proj.weight and
    #        mlp.down_proj.weight with ablate_matrix_output(..., directions[i
    #        + 1], weights[i + 1]), again writing back with copy_()
    raise NotImplementedError
