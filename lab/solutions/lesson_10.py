import contextlib

import torch
import torch.nn.functional as F
from lib.ablate import ablate_vector


def ablation_forward_hook(direction: torch.Tensor):
    d = F.normalize(direction, dim=-1)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        ablated = ablate_vector(hidden, d)
        return (ablated, *output[1:]) if isinstance(output, tuple) else ablated

    return hook


@contextlib.contextmanager
def live_ablation(model, directions: torch.Tensor):
    handles = []
    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(ablation_forward_hook(directions[i + 1])))
        yield model
    finally:
        for handle in handles:
            handle.remove()
