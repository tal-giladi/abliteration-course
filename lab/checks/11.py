import copy
import shutil

import torch
from _lib import check, fail

from exercises.lesson_11 import merge_into_model as student_merge
from lib.ablate import merge_into_model as reference_merge
from lib.common import DEVICE, DTYPE, HARMFUL_TRAIN, HARMLESS_TRAIN, LAB_DIR, encode_chat, load_model, load_tokenizer
from lib.direction import compute_direction, get_residuals_mean
from lib.score import is_refusal

PROMPT = HARMFUL_TRAIN[0]

harmful_mean = get_residuals_mean(HARMFUL_TRAIN)
harmless_mean = get_residuals_mean(HARMLESS_TRAIN)
directions = compute_direction(harmful_mean, harmless_mean)

# --- Part 1: weight-level comparison against the reference, on two
# independent model instances. load_model() is @lru_cache'd — calling it
# twice hands back the SAME object — so we deepcopy the cached singleton
# twice instead, leaving the singleton itself untouched. ---
base_model = load_model()
model_ref = copy.deepcopy(base_model)
model_student = copy.deepcopy(base_model)

reference_merge(model_ref, directions)
try:
    student_merge(model_student, directions)
except NotImplementedError:
    fail("merge_into_model() is still a stub — fill in the TODO in exercises/lesson_11.py")

check(
    torch.allclose(
        model_ref.model.layers[0].self_attn.o_proj.weight,
        model_student.model.layers[0].self_attn.o_proj.weight,
        atol=1e-5,
        rtol=1e-4,
    ),
    "layer 0 self_attn.o_proj.weight matches the reference merge",
)
check(
    torch.allclose(
        model_ref.model.layers[0].mlp.down_proj.weight,
        model_student.model.layers[0].mlp.down_proj.weight,
        atol=1e-5,
        rtol=1e-4,
    ),
    "layer 0 mlp.down_proj.weight matches the reference merge",
)
check(
    not torch.allclose(
        base_model.model.layers[0].self_attn.o_proj.weight, model_ref.model.layers[0].self_attn.o_proj.weight
    ),
    "the merge actually changed the weights (not a no-op)",
)

# --- Part 2: the edit survives a real save/reload cycle. Slower than part 1
# on purpose — a real save_pretrained + from_pretrained round trip through
# disk, not just a tensor comparison — but this course's model is small
# (~0.6B params) and CPU-only, so it's fine to eat that cost here. Reuses
# `model_student`, already merged by the student's own function above. ---
tmp_dir = LAB_DIR / ".cache" / "tmp-lesson-11"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)

try:
    model_student.save_pretrained(tmp_dir)

    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_dir, dtype=DTYPE)
    reloaded.to(DEVICE)
    reloaded.eval()

    tokenizer = load_tokenizer()
    inputs = encode_chat([PROMPT])
    with torch.no_grad():
        out = reloaded.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    check(not is_refusal(response), f"reloaded merged model does not refuse {PROMPT!r}: {response!r}")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)  # lab/.cache/ is gitignored either way

print("Lesson 11: all checks passed.")
