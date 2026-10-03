import torch

from lib import ablate, common, direction, score


def _measure(model, capability_prompts, base_logits=None) -> dict:
    harmful = common.generate(common.HARMFUL_HOLDOUT)
    out = {
        "refusal_lenient": score.refusal_rate(harmful),
        "refusal_strict": score.strict_refusal_rate(harmful),
        "capability": score.capability_spot_check(common.generate(capability_prompts)),
    }
    if base_logits is not None:
        inputs = common.encode_chat(capability_prompts)
        logits = model(**inputs).logits[:, -1, :]
        out["coherence"] = score.coherence_score(logits, base_logits)
    return out


@torch.no_grad()
def abliterate_and_report(model_name: str = common.MODEL_ID) -> dict:
    if model_name != common.MODEL_ID:
        raise ValueError(f"This lab only has {common.MODEL_ID} cached — pass the default model_name.")
    try:
        model = common.load_model()
    except OSError as exc:
        raise RuntimeError(f"{model_name} isn't cached. Run 'bash lab/lab.sh up' first.") from exc

    capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
    inputs = common.encode_chat(capability_prompts)
    base_logits = model(**inputs).logits[:, -1, :]

    before = _measure(model, capability_prompts)

    mean_harmful = direction.get_residuals_mean(common.HARMFUL_TRAIN)
    mean_harmless = direction.get_residuals_mean(common.HARMLESS_TRAIN)
    directions = direction.compute_direction(mean_harmful, mean_harmless)
    ablate.merge_into_model(model, directions, [1.0] * (common.num_layers(model) + 1))

    after = _measure(model, capability_prompts, base_logits=base_logits)
    return {"before": before, "after": after}
