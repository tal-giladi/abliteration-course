import torch
from _lib import check, fail

from exercises.lesson_05 import compute_direction as student_fn
from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN
from lib.direction import compute_direction as reference_fn
from lib.direction import get_residuals_mean

mean_harmful = get_residuals_mean(HARMFUL_TRAIN)
mean_harmless = get_residuals_mean(HARMLESS_TRAIN)

try:
    student_direction = student_fn(mean_harmful, mean_harmless)
except NotImplementedError:
    fail("compute_direction() is still a stub — fill in the TODO in exercises/lesson_05.py")

check(
    tuple(student_direction.shape) == tuple(mean_harmful.shape),
    f"output shape == {tuple(mean_harmful.shape)}",
)

reference_direction = reference_fn(mean_harmful, mean_harmless)
check(
    torch.allclose(student_direction, reference_direction, atol=1e-4, rtol=1e-3),
    "matches the reference implementation",
)

# Layer 0 (the raw embedding output) is a real, expected exception: every
# prompt here is chat-templated with add_generation_prompt=True, so the very
# last token is the same fixed template token for every prompt, regardless of
# what the prompt actually says. At layer 0, before any attention has mixed
# in the earlier tokens, that last-token embedding is therefore IDENTICAL
# across prompts — mean_harmful[0] == mean_harmless[0], the difference is the
# zero vector, and normalize(zero vector) stays zero, not unit length. See
# "Why layer 0 is a zero vector" in the lesson text. Every layer from 1
# onward has seen at least one attention pass and does carry a real signal.
norms = student_direction.norm(dim=-1)
check(
    torch.allclose(norms[1:], torch.ones_like(norms[1:]), atol=1e-4),
    "every layer from 1 onward has a unit-norm direction",
)
check(
    norms[0].item() < 1e-4,
    "layer 0 is the expected zero vector (identical last-token embedding for every prompt)",
)
print("Lesson 05: all checks passed.")
