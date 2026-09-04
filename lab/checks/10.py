import torch
from _lib import check, fail

from exercises.lesson_10 import ablation_forward_hook, live_ablation
from lib.ablate import ablation_forward_hook as reference_hook
from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN, generate, hidden_size, load_model
from lib.direction import compute_direction, get_residuals_mean

torch.manual_seed(0)

# --- Part 1: the hook FUNCTION itself, checked against lib.ablate's
# reference on a synthetic (hidden_states, extra) tuple — no real model
# forward pass needed for this half. ---
direction = torch.randn(hidden_size())

try:
    student_hook = ablation_forward_hook(direction)
except NotImplementedError:
    fail("ablation_forward_hook() is still a stub — fill in the TODO in exercises/lesson_10.py")

reference = reference_hook(direction)

# tuple output, like a real decoder layer returns (hidden_states, past_kv, ...)
hidden_states = torch.randn(2, 5, hidden_size())
extra = torch.zeros(2, 5)
student_out = student_hook(None, None, (hidden_states, extra))
ref_out = reference(None, None, (hidden_states, extra))
check(isinstance(student_out, tuple) and len(student_out) == 2, "hook preserves tuple output shape")
check(
    torch.allclose(student_out[0], ref_out[0], atol=1e-5, rtol=1e-4),
    "ablated hidden states match the reference ablation_forward_hook (tuple output)",
)
check(
    torch.equal(student_out[1], extra),
    "the rest of the output tuple passes through unchanged",
)

# bare-tensor output — not every model/call shape returns a tuple
bare_out = student_hook(None, None, hidden_states)
ref_bare = reference(None, None, hidden_states)
check(
    torch.allclose(bare_out, ref_bare, atol=1e-5, rtol=1e-4),
    "ablated hidden states match the reference ablation_forward_hook (bare tensor output)",
)

# --- Part 2: live_ablation, checked BEHAVIORALLY on a real model — there's
# no single correct context-manager implementation, so this proves the
# observable contract instead: output changes while active, and fully
# reverts once the context manager has exited. ---
PROMPT = HARMFUL_TRAIN[0]

harmful_mean = get_residuals_mean(HARMFUL_TRAIN)
harmless_mean = get_residuals_mean(HARMLESS_TRAIN)
directions = compute_direction(harmful_mean, harmless_mean)

model = load_model()
baseline = generate([PROMPT])[0]

try:
    with live_ablation(model, directions):
        hooked = generate([PROMPT])[0]
except NotImplementedError:
    fail("live_ablation() is still a stub — fill in the TODO in exercises/lesson_10.py")

check(hooked != baseline, "generation changes while live_ablation is active")

after = generate([PROMPT])[0]
check(
    after == baseline,
    "generation reverts to the original output after live_ablation exits — all hooks were removed",
)

print("Lesson 10: all checks passed.")
