import math

from _lib import check, fail

from exercises.lesson_14 import random_search as student_random_search

# A simple 2D bowl, maximized at (3.0, 1.0) with value 0.0 — fixed and known,
# so grading doesn't depend on matching anyone's exact random draw order.
PARAM_SPACE = {"x": (0.0, 6.0), "y": (-2.0, 4.0)}


def objective(x: float, y: float) -> float:
    return -((x - 3.0) ** 2) - ((y - 1.0) ** 2)


try:
    best_params, best_score, history = student_random_search(
        objective, PARAM_SPACE, n_trials=30, seed=0
    )
except NotImplementedError:
    fail("random_search() is still a stub — fill in the TODO in exercises/lesson_14.py")

check(len(history) == 30, f"history has one entry per trial (30), got {len(history)}")

check(
    all(score <= best_score for _, score in history),
    "best_score is >= every score recorded in history",
)

check(
    math.isclose(best_score, max(score for _, score in history), rel_tol=1e-9),
    "best_score exactly matches the best entry recorded in history",
)

check(
    isinstance(best_params, dict) and "x" in best_params and "y" in best_params,
    "best_params contains both search parameters",
)

# The true optimum is (3.0, 1.0), objective 0.0. Over 30 uniform-random trials
# in this 6x6 box, landing within radius ~1.41 of it (score > -2.0) has
# roughly 99% probability for any correct implementation — strict enough to
# prove the search actually worked, generous enough not to be flaky.
check(
    best_score > -2.0,
    f"random_search found a point close to the true optimum (best_score={best_score:.4f}, target > -2.0)",
)

print("Lesson 14: all checks passed.")
