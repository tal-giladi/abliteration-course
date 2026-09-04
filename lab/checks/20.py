from _lib import check, fail

from exercises.lesson_20 import run_mini_heretic

# This one is genuinely slow — a real model load, a real forward pass to
# build the direction from 40 training prompts, an in-place weight merge
# across every layer, and two more rounds of live generation to build the
# report. Expect it to take noticeably longer than any earlier check; on
# this course's small CPU model that's expected, not a bug.

try:
    report = run_mini_heretic()
except NotImplementedError:
    fail("run_mini_heretic() is still a stub — fill in the TODO in exercises/lesson_20.py")
except RuntimeError as exc:
    fail(str(exc))

check(isinstance(report, dict), "run_mini_heretic() returns a dict")
for key in ("refusal_rate", "coherence", "capability"):
    check(key in report, f"report has key {key!r}")

# A loose, honest bar, deliberately: an unmodified, safety-tuned
# Qwen3-0.6B refuses at or near 100% of lib.common.HARMFUL_HOLDOUT (Lesson
# 13's own baseline). We don't assert a precise post-ablation number here —
# it depends on live model output this course can't pin down while writing
# it — only that the merge changed *something* relative to that baseline.
check(report["refusal_rate"] < 1.0, "refusal_rate dropped below the ~1.0 unablated baseline")
check(
    isinstance(report["coherence"], float) and 0.0 <= report["coherence"] <= 1.0,
    "coherence is a float in [0.0, 1.0]",
)
check(
    isinstance(report["capability"], float) and 0.0 <= report["capability"] <= 1.0,
    "capability is a float in [0.0, 1.0]",
)
print("Lesson 20: all checks passed.")
