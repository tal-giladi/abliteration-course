from _lib import check, fail

from exercises.lesson_19 import capability_spot_check as student_fn, run_full_report
from lib.score import CAPABILITY_QUESTIONS, capability_spot_check as reference_fn

# Some hit, some miss — deliberately, so exact equality against the
# reference actually exercises both branches of the substring match.
EXAMPLE_RESPONSES = [
    "Lyon is a nice city, but that's not what you asked.",
    "The answer is four.",
    "You'd get a shade of teal, actually.",
    "Seven, obviously.",
    "It's H2O.",
]

try:
    student_score = student_fn(EXAMPLE_RESPONSES)
except NotImplementedError:
    fail("capability_spot_check() is still a stub — fill in the TODO in exercises/lesson_19.py")

reference_score = reference_fn(EXAMPLE_RESPONSES)
check(student_score == reference_score, f"capability_spot_check(...) == {reference_score}")

check(
    student_fn(EXAMPLE_RESPONSES, CAPABILITY_QUESTIONS) == reference_fn(EXAMPLE_RESPONSES, CAPABILITY_QUESTIONS),
    "capability_spot_check(...) matches the reference with an explicit questions argument too",
)

try:
    report = run_full_report()
except NotImplementedError:
    fail("run_full_report() is still a stub — fill in the TODO in exercises/lesson_19.py")

# Graded on structure, not exact numbers: refusal/coherence/capability all
# depend on the real model's live output, which can't be pinned to a fixed
# expected value at course-authoring time. What's checked is that the shape
# is right and every value is a plausible score.
check(isinstance(report, dict), "run_full_report() returns a dict")
for key in ("refusal_rate", "coherence", "capability"):
    check(key in report, f"report has key {key!r}")
    value = report.get(key)
    check(isinstance(value, float) and 0.0 <= value <= 1.0, f"report[{key!r}] is a float in [0.0, 1.0]")

print("Lesson 19: all checks passed.")
