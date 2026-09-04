import torch
from lib.ablate import ablate_matrix_input, ablate_matrix_output


def merge_into_model(model, directions: torch.Tensor, weights: list[float] | None = None) -> None:
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
