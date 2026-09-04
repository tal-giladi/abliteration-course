from _lib import check, fail

from exercises.lesson_17 import MAPPING

# The 8 real names, confirmed present in heretic's actual src/heretic/model.py and
# src/heretic/main.py, each paired with the toy-course lib/ reference the correct
# answer must name (case-insensitive substring match; any ONE listed keyword is enough
# per key). This is the answer key, built from what was actually read in the real source
# — not guessed.
ANSWER_KEY: dict[str, list[str]] = {
    "Model.abliterate": ["merge_into_model"],
    "AbliterationParameters": ["weights"],
    "get_abliterable_components": ["o_proj", "down_proj"],
    "LoraConfig": ["lora_rank1_factors"],
    "lora_A / lora_B assignment in Model.abliterate": ["lora_rank1_factors"],
    "residual_directions in main.py::run": ["compute_direction"],
    "Model.get_residuals_mean": ["get_residuals_mean"],
    "W = W.to(torch.float32) in Model.abliterate": ["float32"],
}

if not isinstance(MAPPING, dict) or not MAPPING:
    fail("MAPPING is still empty — fill in the TODO in exercises/lesson_17.py")

missing = [key for key in ANSWER_KEY if key not in MAPPING]
missing_msg = "MAPPING has all 8 required keys" if not missing else f"MAPPING has all 8 required keys (missing: {missing})"
check(not missing, missing_msg)

for key, allowed_keywords in ANSWER_KEY.items():
    value = MAPPING.get(key, "")
    check(
        isinstance(value, str) and len(value.strip()) >= 15,
        f"{key!r} has a real explanation (not empty/trivial)",
    )
    lowered = value.lower()
    check(
        any(keyword.lower() in lowered for keyword in allowed_keywords),
        f"{key!r} names the right toy-course equivalent "
        f"(expected one of {allowed_keywords} mentioned in the explanation)",
    )

print("Lesson 17: all checks passed.")
