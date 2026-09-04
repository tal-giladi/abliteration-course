import torch
from _lib import check, fail

from exercises.lesson_08 import ablate_matrix_output as student_output
from exercises.lesson_08 import ablate_matrix_input as student_input
from lib.ablate import ablate_matrix_output as reference_output
from lib.ablate import ablate_matrix_input as reference_input
from lib.common import load_model

torch.manual_seed(0)

direction = torch.randn(6)
# ablate_matrix_output projects `direction` out of W's OUTPUT space, so W's
# out_features (its first dim) must match direction's length, i.e. 6 here —
# exactly like o_proj/down_proj, whose out_features is hidden_size.
W = torch.randn(6, 4)   # synthetic "out_features=6 (== direction), in_features=4"
E = torch.randn(10, 6)  # synthetic "vocab_size=10, hidden_size=6" (rows are residual-stream vectors)

try:
    out_W = student_output(W, direction)
except NotImplementedError:
    fail("ablate_matrix_output() is still a stub — fill in the TODO in exercises/lesson_08.py")

check(
    torch.allclose(out_W, reference_output(W, direction), atol=1e-6),
    "ablate_matrix_output matches lib.ablate on a synthetic matrix",
)
check(
    torch.allclose(
        student_output(W, direction, weight=0.3),
        reference_output(W, direction, weight=0.3),
        atol=1e-6,
    ),
    "ablate_matrix_output respects weight != 1.0",
)

try:
    out_E = student_input(E, direction)
except NotImplementedError:
    fail("ablate_matrix_input() is still a stub — fill in the TODO in exercises/lesson_08.py")

check(
    torch.allclose(out_E, reference_input(E, direction), atol=1e-6),
    "ablate_matrix_input matches lib.ablate on a synthetic matrix",
)

# Real model shapes — no forward pass, just confirms the math holds on the
# actual dimensions Module 04 will edit for real.
model = load_model()
hidden = model.config.hidden_size
real_direction = torch.randn(hidden)

o_proj = model.model.layers[0].self_attn.o_proj.weight
real_out = student_output(o_proj, real_direction)
check(
    tuple(real_out.shape) == tuple(o_proj.shape),
    f"ablate_matrix_output output shape == {tuple(o_proj.shape)} on real o_proj.weight",
)
check(
    torch.allclose(real_out, reference_output(o_proj, real_direction), atol=1e-4, rtol=1e-3),
    "ablate_matrix_output matches lib.ablate on real o_proj.weight",
)

embed = model.model.embed_tokens.weight
real_E = student_input(embed, real_direction)
check(
    tuple(real_E.shape) == tuple(embed.shape),
    f"ablate_matrix_input output shape == {tuple(embed.shape)} on real embed_tokens.weight",
)
check(
    torch.allclose(real_E, reference_input(embed, real_direction), atol=1e-4, rtol=1e-3),
    "ablate_matrix_input matches lib.ablate on real embed_tokens.weight",
)

print("Lesson 08: all checks passed.")
