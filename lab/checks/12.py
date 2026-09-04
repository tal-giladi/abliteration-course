import math

from _lib import check, fail

from exercises.lesson_12 import strength_curve

try:
    result1 = strength_curve(4, 1.0, 0.5, 0.0, 0.5)
except NotImplementedError:
    fail("strength_curve() is still a stub — fill in the TODO in exercises/lesson_12.py")

# num_layers=4, peak at position 0.5, floor 0.0: positions 0,.25,.5,.75,1;
# dist from 0.5 = .5,.25,0,.25,.5; t = dist / 0.5 = 1,.5,0,.5,1 (no clamping
# needed); weight = 1.0 - 1.0 * t
expected1 = [0.0, 0.5, 1.0, 0.5, 0.0]
check(len(result1) == 5, "strength_curve(4, ...) returns 5 values (num_layers + 1)")
check(
    all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(result1, expected1)),
    f"strength_curve(4, 1.0, 0.5, 0.0, 0.5) == {expected1}",
)

# num_layers=4, peak at the very start (position 0.0), narrow falloff:
# positions 0,.25,.5,.75,1; dist = 0,.25,.5,.75,1;
# t = dist / 0.25 = 0,1,2,3,4 -> clamped to <=1 -> 0,1,1,1,1;
# weight = 1.0 - 0.8 * t
result2 = strength_curve(4, 1.0, 0.0, 0.2, 0.25)
expected2 = [1.0, 0.2, 0.2, 0.2, 0.2]
check(
    all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(result2, expected2)),
    f"strength_curve(4, 1.0, 0.0, 0.2, 0.25) == {expected2}",
)

# num_layers=2, peak at the very end (position 1.0): positions 0,.5,1;
# dist from 1.0 = 1,.5,0; t = dist / 1.0 = 1,.5,0; weight = 0.8 - 0.7 * t
result3 = strength_curve(2, 0.8, 1.0, 0.1, 1.0)
expected3 = [0.1, 0.45, 0.8]
check(len(result3) == 3, "strength_curve(2, ...) returns 3 values (num_layers + 1)")
check(
    all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(result3, expected3)),
    f"strength_curve(2, 0.8, 1.0, 0.1, 1.0) == {expected3}",
)

# num_layers=6, peak in the middle, narrow min_weight_distance so most
# checkpoints are clamped at the floor: positions 0,1/6,2/6,.5,4/6,5/6,1;
# dist from 0.5 = .5, 1/3, 1/6, 0, 1/6, 1/3, .5; t = dist / 0.2 clamped to
# <=1 -> 1, 1, 1/6/0.2 (=0.8333...), 0, 0.8333..., 1, 1;
# weight = 1.0 - 0.7 * t
result4 = strength_curve(6, 1.0, 0.5, 0.3, 0.2)
mid_weight = 1.0 - 0.7 * ((1 / 6) / 0.2)
expected4 = [0.3, 0.3, mid_weight, 1.0, mid_weight, 0.3, 0.3]
check(len(result4) == 7, "strength_curve(6, ...) returns 7 values (num_layers + 1)")
check(
    all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(result4, expected4)),
    f"strength_curve(6, 1.0, 0.5, 0.3, 0.2) == {expected4}",
)

print("Lesson 12: all checks passed.")
