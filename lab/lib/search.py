# SPDX-License-Identifier: MIT
"""A deliberately dumb search loop: uniform random sampling. Lesson 14 has
you beat a fixed baseline with this; Lesson 15 reads the real thing Heretic
uses instead (Optuna's TPE sampler) and explains what it does that random
search doesn't.
"""

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
    rng = random.Random(seed)
    best_params, best_score = None, float("-inf")
    history = []
    for _ in range(n_trials):
        params = {name: rng.uniform(lo, hi) for name, (lo, hi) in param_space.items()}
        score = objective(**params)
        history.append((params, score))
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score, history
