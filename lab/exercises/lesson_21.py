# Lesson 21 (extension) - Reproducing a published evaluation: one number is never the number
# Read lessons/module-07/lesson-03.md before filling this in.

from lib import common, score


def strict_refusal_rate(responses: list[str]) -> float:
    """From-scratch reimplementation of lib.score.strict_refusal_rate.

    A response counts as a refusal if it is an explicit refusal
    (lib.score.is_refusal) OR contains any phrase in
    lib.score.SOFT_REFUSAL_MARKERS (case-insensitive substring). Return the
    fraction of responses that count. By construction this is always >= the
    lenient lib.score.refusal_rate on the same responses.
    """
    # TODO: for each response, mark it a refusal if score.is_refusal(response)
    # OR any SOFT_REFUSAL_MARKER is a substring of response.lower(); return the
    # fraction that are refusals (0.0 for an empty list).
    raise NotImplementedError


def compare_scorers(responses: list[str]) -> dict:
    """Return {"lenient": ..., "strict": ..., "gap": strict - lenient}, where
    lenient is lib.score.refusal_rate and strict is YOUR strict_refusal_rate
    above. `gap` is a single number for "how much does my reported refusal
    rate depend on which scorer I picked" — the whole point of this lesson.
    """
    # TODO:
    # 1. lenient = score.refusal_rate(responses)
    # 2. strict = strict_refusal_rate(responses)
    # 3. return {"lenient": lenient, "strict": strict, "gap": strict - lenient}
    raise NotImplementedError
