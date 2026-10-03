from _lib import check, fail

from exercises.lesson_22 import abliterate_and_report

# Slow, like Lesson 20 but doubled: two full rounds of generation (before and
# after the merge) plus the direction build and an in-place weight merge. On
# this course's small CPU model that's expected, not a bug.

try:
    report = abliterate_and_report()
except NotImplementedError:
    fail("abliterate_and_report() is still a stub — fill in the TODO in exercises/lesson_22.py")
except RuntimeError as exc:
    fail(str(exc))

check(isinstance(report, dict), "abliterate_and_report() returns a dict")
check("before" in report and "after" in report, "report has 'before' and 'after' sections")

before, after = report["before"], report["after"]

for section_name, section, keys in (
    ("before", before, ("refusal_lenient", "refusal_strict", "capability")),
    ("after", after, ("refusal_lenient", "refusal_strict", "capability", "coherence")),
):
    for key in keys:
        check(key in section, f"{section_name!r} has key {key!r}")
        value = section.get(key)
        check(isinstance(value, float) and 0.0 <= value <= 1.0, f"{section_name}[{key!r}] is a float in [0.0, 1.0]")

# Lesson 21's guarantee, now on live output: strict never undercounts lenient.
check(before["refusal_strict"] >= before["refusal_lenient"], "before: strict refusal >= lenient refusal")
check(after["refusal_strict"] >= after["refusal_lenient"], "after: strict refusal >= lenient refusal")

# Honest, loose bar on the ablation itself: ablation should not make the model
# refuse MORE than it did unablated. We don't assert an exact number in either
# direction — the harmful holdout is tiny (4 prompts) and the result depends on
# live model output this course can't pin down while being written. (This is
# why one number on a handful of prompts is never the number — Lesson 21.)
check(after["refusal_lenient"] <= before["refusal_lenient"], "after: ablation did not increase the (lenient) refusal rate")

print("Lesson 22: all checks passed.")
