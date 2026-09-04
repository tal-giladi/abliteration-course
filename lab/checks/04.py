import torch
from _lib import check, fail

from exercises.lesson_04 import get_residuals_mean as student_fn
from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN, hidden_size, num_layers
from lib.direction import get_residuals_mean as reference_fn

try:
    student_harmful = student_fn(HARMFUL_TRAIN)
except NotImplementedError:
    fail("get_residuals_mean() is still a stub — fill in the TODO in exercises/lesson_04.py")

expected_shape = (num_layers() + 1, hidden_size())
check(tuple(student_harmful.shape) == expected_shape, f"output shape == {expected_shape}")

reference_harmful = reference_fn(HARMFUL_TRAIN)
check(
    torch.allclose(student_harmful, reference_harmful, atol=1e-4, rtol=1e-3),
    "matches the reference implementation on HARMFUL_TRAIN",
)

student_harmless = student_fn(HARMLESS_TRAIN)
reference_harmless = reference_fn(HARMLESS_TRAIN)
check(
    torch.allclose(student_harmless, reference_harmless, atol=1e-4, rtol=1e-3),
    "matches the reference implementation on HARMLESS_TRAIN",
)

# The whole point of this module: the two means had better not be the same
# vector at every layer, or there is nothing for Lesson 05 to build a
# direction out of. Loose, structural threshold — chosen without being able
# to empirically tune it on this machine, so it's deliberately generous.
diffs = (student_harmful - student_harmless).norm(dim=-1)  # [num_layers + 1]
check(
    bool((diffs > 0.01).any().item()),
    "harmful-mean and harmless-mean differ by more than noise at some layer",
)
print("Lesson 04: all checks passed.")
