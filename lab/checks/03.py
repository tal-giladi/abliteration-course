import math

import torch
from _lib import check, fail

from exercises.lesson_03 import cosine_similarity, mean_difference

a = torch.tensor([3.0, 4.0])
b = torch.tensor([6.0, 8.0])
c = torch.tensor([-4.0, 3.0])

try:
    sim_same = cosine_similarity(a, b)
except NotImplementedError:
    fail("cosine_similarity() is still a stub — fill in the TODO in exercises/lesson_03.py")

check(math.isclose(sim_same, 1.0, abs_tol=1e-6), "cosine_similarity(a, b) == 1.0 for same-direction vectors")

sim_perp = cosine_similarity(a, c)
check(math.isclose(sim_perp, 0.0, abs_tol=1e-6), "cosine_similarity(a, c) == 0.0 for perpendicular vectors")

sim_opposite = cosine_similarity(a, -a)
check(math.isclose(sim_opposite, -1.0, abs_tol=1e-6), "cosine_similarity(a, -a) == -1.0 for opposite vectors")

try:
    diff = mean_difference(torch.tensor([[1.0, 1.0], [3.0, 3.0]]), torch.tensor([[0.0, 0.0], [0.0, 2.0]]))
except NotImplementedError:
    fail("mean_difference() is still a stub — fill in the TODO in exercises/lesson_03.py")

expected = torch.tensor([2.0, 1.0])  # mean([1,1],[3,3])=[2,2]; mean([0,0],[0,2])=[0,1]; diff=[2,1]
check(torch.allclose(diff, expected, atol=1e-6), f"mean_difference(...) == {expected.tolist()}")
print("Lesson 03: all checks passed.")
