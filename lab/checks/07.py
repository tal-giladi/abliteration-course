import torch
from _lib import check, fail

from exercises.lesson_07 import ablate_vector as student_fn
from lib.ablate import ablate_vector as reference_fn

torch.manual_seed(0)

direction = torch.tensor([3.0, 4.0, 0.0])  # not unit length on purpose — must be normalized internally
d_unit = direction / direction.norm()

v_single = torch.tensor([1.0, 2.0, 5.0])

try:
    out_single = student_fn(v_single, direction)
except NotImplementedError:
    fail("ablate_vector() is still a stub — fill in the TODO in exercises/lesson_07.py")

check(
    torch.allclose(out_single, reference_fn(v_single, direction), atol=1e-6),
    "matches lib.ablate.ablate_vector for a single vector",
)
check(
    torch.allclose(out_single @ d_unit, torch.tensor(0.0), atol=1e-5),
    "result is orthogonal to the direction (single vector)",
)

v_batch = torch.randn(5, 3)
out_batch = student_fn(v_batch, direction)
check(
    torch.allclose(out_batch, reference_fn(v_batch, direction), atol=1e-6),
    "matches lib.ablate.ablate_vector for a batch",
)
check(
    torch.allclose(out_batch @ d_unit, torch.zeros(5), atol=1e-5),
    "every row of the result is orthogonal to the direction (batch)",
)
print("Lesson 07: all checks passed.")
