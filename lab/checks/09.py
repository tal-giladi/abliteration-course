import torch
from _lib import check, fail

from exercises.lesson_09 import lora_rank1_factors as student_fn
from lib.ablate import ablate_matrix_output
from lib.ablate import lora_rank1_factors as reference_fn
from lib.common import load_model

torch.manual_seed(0)

direction = torch.randn(6)
# out_features (W's first dim) must match direction's length — see checks/08.py.
W = torch.randn(6, 4)

try:
    A, B = student_fn(W, direction)
except NotImplementedError:
    fail("lora_rank1_factors() is still a stub — fill in the TODO in exercises/lesson_09.py")

ref_A, ref_B = reference_fn(W, direction)

check(tuple(A.shape) == (1, W.shape[1]), f"A has shape (1, {W.shape[1]})")
check(tuple(B.shape) == (W.shape[0], 1), f"B has shape ({W.shape[0]}, 1)")
check(torch.allclose(A, ref_A, atol=1e-6), "A matches lib.ablate.lora_rank1_factors")
check(torch.allclose(B, ref_B, atol=1e-6), "B matches lib.ablate.lora_rank1_factors")

# The proof: reconstructing W from the STUDENT's own A/B must equal the
# direct projection edit, exactly — not just "close to the reference output".
expected = ablate_matrix_output(W, direction, weight=1.0)
check(
    torch.allclose(W + B @ A, expected, atol=1e-5),
    "W + B @ A == ablate_matrix_output(W, direction) — the LoRA delta is exact, not approximate",
)

# Same proof again on the real o_proj weight shape.
model = load_model()
hidden = model.config.hidden_size
o_proj = model.model.layers[0].self_attn.o_proj.weight
real_direction = torch.randn(hidden)

A_real, B_real = student_fn(o_proj, real_direction)
check(
    tuple(A_real.shape) == (1, o_proj.shape[1]),
    f"A has shape (1, {o_proj.shape[1]}) on real o_proj.weight",
)
check(
    tuple(B_real.shape) == (o_proj.shape[0], 1),
    f"B has shape ({o_proj.shape[0]}, 1) on real o_proj.weight",
)
check(
    torch.allclose(
        o_proj + B_real @ A_real,
        ablate_matrix_output(o_proj, real_direction, weight=1.0),
        atol=1e-4,
        rtol=1e-3,
    ),
    "W + B @ A == ablate_matrix_output(W, direction) on real o_proj.weight",
)

print("Lesson 09: all checks passed.")
