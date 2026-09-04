# Lesson 14 - Search instead of guesswork
# Read lessons/module-05/lesson-02.md before filling this in.

import random
from typing import Callable


def random_search(
    objective: Callable[..., float],
    param_space: dict[str, tuple[float, float]],
    n_trials: int = 15,
    seed: int = 0,
):
    """Maximize `objective(**params)` by uniform-random sampling.

    param_space: {name: (low, high)} — each trial draws one uniform value
    per parameter and calls objective(name=value, ...).
    Returns (best_params, best_score, history) where history is a list of
    (params, score) for every trial, in order.
    """
    # TODO: seed a random.Random(seed) once, outside the loop. For each of
    # n_trials trials, draw one uniform value per parameter from param_space,
    # call objective(**params), append (params, score) to history, and keep
    # whichever (params, score) has the highest score seen so far.
    raise NotImplementedError
