import torch
from _lib import check, fail

from exercises.lesson_02 import get_residuals_per_prompt as student_fn
from lib.common import HARMLESS_PROMPTS, hidden_size, num_layers
from lib.direction import get_residuals_per_prompt as reference_fn

PROMPTS = HARMLESS_PROMPTS[:3]

try:
    student_out = student_fn(PROMPTS)
except NotImplementedError:
    fail("get_residuals_per_prompt() is still a stub — fill in the TODO in exercises/lesson_02.py")

expected_shape = (len(PROMPTS), num_layers() + 1, hidden_size())
check(tuple(student_out.shape) == expected_shape, f"output shape == {expected_shape}")

reference_out = reference_fn(PROMPTS)
check(
    torch.allclose(student_out, reference_out, atol=1e-4, rtol=1e-3),
    "matches the reference implementation on real model output",
)
print("Lesson 02: all checks passed.")
