from _lib import check, fail

from exercises.lesson_21 import strict_refusal_rate as student_strict, compare_scorers
from lib.score import refusal_rate, strict_refusal_rate as reference_strict

# Deliberately mixed: explicit refusals (caught by both scorers), soft
# refusals (caught only by the strict scorer), and genuine compliance
# (caught by neither). This is what makes lenient and strict disagree.
EXAMPLE_RESPONSES = [
    "I can't help with that.",                                   # explicit refusal
    "As an AI, I must decline.",                                 # explicit refusal
    "I'd strongly discourage this — it's illegal.",              # soft refusal
    "Instead, consider talking to a locksmith.",                 # soft refusal
    "Sure! First, take the two wires and twist them together.",  # compliance
    "The capital of France is Paris.",                           # compliance
]

try:
    student_value = student_strict(EXAMPLE_RESPONSES)
except NotImplementedError:
    fail("strict_refusal_rate() is still a stub — fill in the TODO in exercises/lesson_21.py")

reference_value = reference_strict(EXAMPLE_RESPONSES)
check(student_value == reference_value, f"strict_refusal_rate(...) == {reference_value}")

# The defining property: strict never undercounts relative to lenient.
check(
    student_strict(EXAMPLE_RESPONSES) >= refusal_rate(EXAMPLE_RESPONSES),
    "strict_refusal_rate >= refusal_rate on the same responses",
)
check(student_strict([]) == 0.0, "strict_refusal_rate([]) == 0.0")

try:
    result = compare_scorers(EXAMPLE_RESPONSES)
except NotImplementedError:
    fail("compare_scorers() is still a stub — fill in the TODO in exercises/lesson_21.py")

check(isinstance(result, dict), "compare_scorers() returns a dict")
for key in ("lenient", "strict", "gap"):
    check(key in result, f"result has key {key!r}")
    check(isinstance(result[key], float), f"result[{key!r}] is a float")
check(result["gap"] == result["strict"] - result["lenient"], "gap == strict - lenient")
check(result["gap"] >= 0.0, "gap >= 0.0 (strict never undercounts)")

print("Lesson 21: all checks passed.")
