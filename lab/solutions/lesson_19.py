from lib import common, score


def capability_spot_check(
    responses: list[str],
    questions: list[tuple[str, list[str]]] = score.CAPABILITY_QUESTIONS,
) -> float:
    if not responses:
        return 0.0
    hits = 0
    for response, (_prompt, acceptable) in zip(responses, questions):
        lowered = response.lower()
        if any(answer in lowered for answer in acceptable):
            hits += 1
    return hits / len(questions)


def run_full_report(model=None, base_logits=None) -> dict:
    model = model or common.load_model()

    harmful_responses = common.generate(common.HARMFUL_HOLDOUT)
    refusal = score.refusal_rate(harmful_responses)

    capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
    capability_responses = common.generate(capability_prompts)
    capability = capability_spot_check(capability_responses)

    if base_logits is not None:
        inputs = common.encode_chat(capability_prompts)
        edited_logits = model(**inputs).logits[:, -1, :]
        coherence = score.coherence_score(edited_logits, base_logits)
    else:
        coherence = 1.0

    return score.full_report(refusal, coherence, capability)
