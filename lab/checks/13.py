import math

import torch
from _lib import check, fail

from exercises.lesson_13 import coherence_score, is_refusal, refusal_rate
from lib.score import coherence_score as reference_coherence_score
from lib.score import is_refusal as reference_is_refusal
from lib.score import refusal_rate as reference_refusal_rate

RESPONSES = [
    "I can't help with that request.",
    "Sure — here's how you do it: first, gather your ingredients.",
    "I'm sorry, but I cannot assist with that.",
    "The capital of France is Paris.",
    "As an AI, I am not able to provide that information.",
    "Sourdough bread needs a starter, flour, water, and salt.",
    "I must decline to answer that one.",
    "Sure, here's a haiku about autumn for you.",
]

try:
    student_flags = [is_refusal(r) for r in RESPONSES]
except NotImplementedError:
    fail("is_refusal() is still a stub — fill in the TODO in exercises/lesson_13.py")

reference_flags = [reference_is_refusal(r) for r in RESPONSES]
check(
    student_flags == reference_flags,
    "is_refusal(...) matches lib.score.is_refusal on a mixed set of responses",
)

try:
    student_rate = refusal_rate(RESPONSES)
except NotImplementedError:
    fail("refusal_rate() is still a stub — fill in the TODO in exercises/lesson_13.py")

reference_rate = reference_refusal_rate(RESPONSES)
check(
    math.isclose(student_rate, reference_rate, abs_tol=1e-9),
    f"refusal_rate(...) == {reference_rate}",
)
check(
    math.isclose(refusal_rate([]), 0.0, abs_tol=1e-9),
    "refusal_rate([]) == 0.0 (no responses, no refusals)",
)

torch.manual_seed(0)
identical_logits = torch.randn(4, 50)

try:
    identical_score = coherence_score(identical_logits, identical_logits)
except NotImplementedError:
    fail("coherence_score() is still a stub — fill in the TODO in exercises/lesson_13.py")

check(
    math.isclose(identical_score, 1.0, abs_tol=1e-4),
    f"coherence_score(x, x) ~= 1.0 for identical logits, got {identical_score}",
)

# Two near-one-hot distributions concentrated on DIFFERENT tokens — about as
# different as two next-token distributions can get.
base_logits = torch.zeros(4, 50)
base_logits[:, 0] = 20.0
edited_logits = torch.zeros(4, 50)
edited_logits[:, 1] = 20.0

different_score = coherence_score(edited_logits, base_logits)
check(
    different_score < 0.01,
    f"coherence_score is much lower than 1.0 for very different distributions, got {different_score}",
)

reference_identical = reference_coherence_score(identical_logits, identical_logits)
reference_different = reference_coherence_score(edited_logits, base_logits)
check(
    math.isclose(identical_score, reference_identical, abs_tol=1e-4),
    "coherence_score(x, x) matches lib.score.coherence_score",
)
check(
    math.isclose(different_score, reference_different, abs_tol=1e-4),
    "coherence_score(very different logits) matches lib.score.coherence_score",
)

print("Lesson 13: all checks passed.")
