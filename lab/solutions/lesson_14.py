import random
from typing import Callable


def random_search(
    objective: Callable[..., float],
    param_space: dict[str, tuple[float, float]],
    n_trials: int = 15,
    seed: int = 0,
):
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
