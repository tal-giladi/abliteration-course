import torch
from lib.common import encode_chat, load_model


@torch.no_grad()
def get_residuals_per_prompt(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    model = load_model()
    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = encode_chat(batch)
        out = model(**inputs, output_hidden_states=True)
        stacked = torch.stack(out.hidden_states, dim=1)
        rows.append(stacked[:, :, -1, :])
    return torch.cat(rows, dim=0)
