from lib.common import encode_chat, load_model


def inspect_model(prompt: str) -> dict:
    model = load_model()
    inputs = encode_chat([prompt])
    out = model(**inputs, output_hidden_states=True)
    final_stream = out.hidden_states[-1][:, -1, :]
    return {
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "final_norm": final_stream.norm().item(),
    }
