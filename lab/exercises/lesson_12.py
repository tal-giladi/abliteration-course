# Lesson 12 - Choosing which layers and how strongly
# Read lessons/module-04/lesson-03.md before filling this in.
# No model, no torch needed here — this is pure per-index arithmetic.


def strength_curve(
    num_layers: int,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> list[float]:
    """One ablation-strength multiplier per residual-stream checkpoint —
    length num_layers + 1, exactly the shape merge_into_model's `weights`
    argument expects (Lesson 11).

    `max_weight_position` is a fraction in [0, 1] of the way along the layer
    stack (0 = the embedding output, 1 = the last layer) where strength
    peaks at `max_weight`. Moving away from that position, in units of
    `min_weight_distance` (also a fraction of the stack), strength falls off
    LINEARLY toward `min_weight` and is floored there — it never goes below
    `min_weight`, no matter how far from the peak a checkpoint is.
    """
    # TODO: for each checkpoint i in range(num_layers + 1):
    #  - pos = i / num_layers                        (float division!)
    #  - dist = abs(pos - max_weight_position)
    #  - t = min(dist / min_weight_distance, 1.0)     (guard division by
    #    zero: if min_weight_distance <= 0, t is 0.0 exactly at the peak
    #    and 1.0 everywhere else)
    #  - weight = max_weight - (max_weight - min_weight) * t
    raise NotImplementedError
