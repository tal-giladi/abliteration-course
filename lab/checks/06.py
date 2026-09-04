import torch
from _lib import check, fail

from exercises.lesson_06 import projection_scores as student_fn
from lib.common import HARMFUL_HOLDOUT, HARMFUL_TRAIN, HARMLESS_HOLDOUT, HARMLESS_TRAIN, num_layers
from lib.direction import compute_direction, get_residuals_mean, get_residuals_per_prompt
from lib.direction import projection_scores as reference_fn

# --- Part 1: matches the reference on a small, fixed input ---
torch.manual_seed(0)
n_layers = num_layers() + 1
fake_residuals = torch.randn(5, n_layers, 8)
fake_direction = torch.randn(n_layers, 8)
LAYER = n_layers // 2

try:
    student_scores = student_fn(fake_residuals, fake_direction, LAYER)
except NotImplementedError:
    fail("projection_scores() is still a stub — fill in the TODO in exercises/lesson_06.py")

check(tuple(student_scores.shape) == (5,), "output shape == (num_prompts,)")

reference_scores = reference_fn(fake_residuals, fake_direction, LAYER)
check(
    torch.allclose(student_scores, reference_scores, atol=1e-4, rtol=1e-3),
    "matches the reference implementation on a fixed input",
)

# --- Part 2: the real proof — separation on held-out prompts, real model ---
mean_harmful = get_residuals_mean(HARMFUL_TRAIN)
mean_harmless = get_residuals_mean(HARMLESS_TRAIN)
direction = compute_direction(mean_harmful, mean_harmless)

harmful_residuals = get_residuals_per_prompt(HARMFUL_HOLDOUT)
harmless_residuals = get_residuals_per_prompt(HARMLESS_HOLDOUT)

total_layers = num_layers() + 1
candidate_layers = sorted({total_layers // 4, total_layers // 2, (3 * total_layers) // 4})

separated = False
for layer in candidate_layers:
    harmful_scores = student_fn(harmful_residuals, direction, layer)
    harmless_scores = student_fn(harmless_residuals, direction, layer)
    if harmful_scores.mean().item() > harmless_scores.mean().item():
        separated = True
        break

check(
    separated,
    "at at least one candidate middle layer, held-out harmful prompts score higher than held-out harmless prompts",
)
print("Lesson 06: all checks passed.")
