# Lesson 10 - Hooking every layer at once
# Read lessons/module-04/lesson-01.md before filling this in.

import contextlib

import torch
import torch.nn.functional as F


def ablation_forward_hook(direction: torch.Tensor):
    """A forward hook that ablates a decoder layer's *output* live, without
    touching any weights. Register on `model.model.layers[i]`.

    `direction` is the [hidden_size] direction to remove. The returned
    `hook(module, inputs, output)` must:
      - pull the hidden-states tensor out of `output` (a decoder layer's
        output is often a tuple whose first element is the hidden states and
        whose remaining elements — attention weights, a cache entry — must
        be passed through unchanged)
      - remove `direction`'s component from it (Lesson 07's operation —
        `lib.ablate.ablate_vector` is exactly this, reuse it)
      - return the result in the SAME shape `output` came in as (a tuple in,
        a tuple out; a bare tensor in, a bare tensor out)
    """
    # TODO: normalize `direction` once, outside the inner hook function, then
    # define and return `hook(module, inputs, output)` per the docstring.
    raise NotImplementedError


@contextlib.contextmanager
def live_ablation(model, directions: torch.Tensor):
    """Context manager: registers `ablation_forward_hook(directions[i + 1])`
    on every `model.model.layers[i]` on enter, and removes every hook it
    registered on exit — including when the caller's code raises.

    directions: [num_layers + 1, hidden_size], e.g. from
    lib.direction.compute_direction(). Layer i gets directions[i + 1]
    because index 0 is the embedding output, before any decoder layer runs.
    """
    # TODO: register a hook (register_forward_hook) on each layer, keep the
    # returned handles, `yield`, then remove every handle. Put the
    # registration + yield in a `try` and the removal in a `finally` so
    # cleanup happens even if the `with` block raises.
    raise NotImplementedError
