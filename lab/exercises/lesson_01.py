# Lesson 01 - Tokens, embeddings, and the residual stream
# Read lessons/module-01/lesson-01.md before filling this in.

from lib.common import encode_chat, load_model


def inspect_model(prompt: str) -> dict:
    """Return {"hidden_size": int, "num_layers": int, "final_norm": float}
    for a single prompt, where final_norm is the L2 norm of the residual
    stream at the LAST layer, LAST token.
    """
    # TODO: load the model, run the prompt with output_hidden_states=True,
    # and read off the three values from model.config and the last
    # hidden_states entry.
    raise NotImplementedError
