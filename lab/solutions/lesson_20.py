import torch

from lib import ablate, common, direction, score


@torch.no_grad()
def run_mini_heretic(model_name: str = common.MODEL_ID) -> dict:
    if model_name != common.MODEL_ID:
        raise ValueError(f"This lab only has {common.MODEL_ID} cached — pass the default model_name.")

    try:
        model = common.load_model()
    except OSError as exc:
        raise RuntimeError(f"{model_name} isn't cached. Run 'bash lab/lab.sh up' first.") from exc

    capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
    inputs = common.encode_chat(capability_prompts)
    base_logits = model(**inputs).logits[:, -1, :]

    mean_harmful = direction.get_residuals_mean(common.HARMFUL_TRAIN)
    mean_harmless = direction.get_residuals_mean(common.HARMLESS_TRAIN)
    directions = direction.compute_direction(mean_harmful, mean_harmless)

    weights = [1.0] * (common.num_layers(model) + 1)
    ablate.merge_into_model(model, directions, weights)

    harmful_responses = common.generate(common.HARMFUL_HOLDOUT)
    refusal = score.refusal_rate(harmful_responses)

    capability_responses = common.generate(capability_prompts)
    capability = score.capability_spot_check(capability_responses)

    inputs = common.encode_chat(capability_prompts)
    edited_logits = model(**inputs).logits[:, -1, :]
    coherence = score.coherence_score(edited_logits, base_logits)

    return score.full_report(refusal, coherence, capability)
