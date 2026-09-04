# Lesson 02 - Reading hidden states across layers
# Read lessons/module-01/lesson-02.md before filling this in.

import torch
from lib.common import encode_chat, load_model


@torch.no_grad()
def get_residuals_per_prompt(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    """Last-token residual stream for every prompt, at every layer.
    Returns a Tensor [len(prompts), num_layers + 1, hidden_size].
    """
    # TODO: batch through `prompts`, call encode_chat + the model with
    # output_hidden_states=True, stack the hidden_states tuple into one
    # tensor, slice out the last token, and concatenate across batches.
    raise NotImplementedError
