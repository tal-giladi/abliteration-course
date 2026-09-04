# Lesson 19 - Evaluating a decensored model properly
# Read lessons/module-07/lesson-01.md before filling this in.

from lib import common, score


def capability_spot_check(
    responses: list[str],
    questions: list[tuple[str, list[str]]] = score.CAPABILITY_QUESTIONS,
) -> float:
    """responses: model answers to `questions`'s prompts, in the same order.
    Returns the fraction whose response contains an acceptable answer
    (case-insensitive substring match). A from-scratch reimplementation of
    lib.score.capability_spot_check — identical signature, identical
    behavior; the checker compares your output to that reference directly.
    """
    # TODO: for each (response, (_prompt, acceptable)) pair, check whether any
    # acceptable answer appears (case-insensitively) as a substring of the
    # response; return the fraction of questions that hit.
    raise NotImplementedError


def run_full_report(model=None, base_logits=None) -> dict:
    """Generate responses to lib.common.HARMFUL_HOLDOUT and score
    lib.score.refusal_rate; generate responses to the 5 prompts in
    lib.score.CAPABILITY_QUESTIONS and score them with YOUR
    capability_spot_check above; assemble both, plus a coherence score,
    into lib.score.full_report(...).

    Coherence needs a stored "before" copy of the unablated model's logits
    to compare against (see Lesson 11 — once a model's weights are merged
    in place, the unablated version is gone unless something was saved
    first). This function accepts that as an optional `base_logits`
    argument — shape [len(CAPABILITY_QUESTIONS), vocab_size], one row per
    capability-question prompt, computed by the CALLER before any ablation
    happens. If it isn't supplied, report coherence as 1.0 — a placeholder
    meaning "not measured this run," not "identical to the base model."
    """
    # TODO:
    # 1. model = model or common.load_model()
    # 2. responses = common.generate(common.HARMFUL_HOLDOUT)
    #    refusal = score.refusal_rate(responses)
    # 3. capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
    #    capability_responses = common.generate(capability_prompts)
    #    capability = capability_spot_check(capability_responses)
    # 4. if base_logits is not None: forward capability_prompts through
    #    `model`, take last-token logits, score.coherence_score(edited_logits,
    #    base_logits); else coherence = 1.0
    # 5. return score.full_report(refusal, coherence, capability)
    raise NotImplementedError
