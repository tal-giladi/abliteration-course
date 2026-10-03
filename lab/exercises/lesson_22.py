# Lesson 22 (extension) - The full recipe, measured like the paper (and where the paper stops)
# Read lessons/module-07/lesson-04.md before filling this in.

import torch

from lib import ablate, common, direction, score


def abliterate_and_report(model_name: str = common.MODEL_ID) -> dict:
    """Run the whole abliteration pipeline on the weak lab model and return a
    before/after report in the shape a published study uses:

        {
          "before": {"refusal_lenient": f, "refusal_strict": f, "capability": f},
          "after":  {"refusal_lenient": f, "refusal_strict": f, "capability": f, "coherence": f},
        }

    Follow the seven steps in lessons/module-07/lesson-04.md. The ordering that
    matters: capture base_logits and the full "before" measurement on the
    UNABLATED model (Step 2), THEN compute the direction and merge (Steps 3-5),
    THEN measure "after" including coherence against base_logits (Step 6).
    Guard model_name and the missing-cache OSError exactly as Lesson 20 did.
    """
    # TODO:
    # 1. if model_name != common.MODEL_ID: raise ValueError(...)
    # 2. try: model = common.load_model(); except OSError: raise RuntimeError(... 'bash lab/lab.sh up' ...)
    # 3. capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
    #    base_logits = model(**common.encode_chat(capability_prompts)).logits[:, -1, :]
    # 4. before = measure refusal_rate + strict_refusal_rate on common.HARMFUL_HOLDOUT,
    #    and capability_spot_check on capability_prompts (NO coherence yet)
    # 5. direction from HARMFUL_TRAIN/HARMLESS_TRAIN; ablate.merge_into_model(
    #    model, directions, [1.0] * (common.num_layers(model) + 1))
    # 6. after = same three measurements, PLUS coherence_score(after_logits, base_logits)
    # 7. return {"before": before, "after": after}
    # Decorate the function with @torch.no_grad() (add it above the def).
    raise NotImplementedError
