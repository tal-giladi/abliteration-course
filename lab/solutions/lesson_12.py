def strength_curve(
    num_layers: int,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> list[float]:
    curve = []
    for i in range(num_layers + 1):
        pos = i / num_layers
        dist = abs(pos - max_weight_position)
        if min_weight_distance <= 0:
            t = 0.0 if dist == 0 else 1.0
        else:
            t = min(dist / min_weight_distance, 1.0)
        curve.append(max_weight - (max_weight - min_weight) * t)
    return curve
