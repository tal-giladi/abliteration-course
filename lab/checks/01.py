from _lib import check, fail

from exercises.lesson_01 import inspect_model
from lib.common import load_model

PROMPT = "What is the capital of France?"

try:
    result = inspect_model(PROMPT)
except NotImplementedError:
    fail("inspect_model() is still a stub — fill in the TODO in exercises/lesson_01.py")

model = load_model()

check(
    result.get("hidden_size") == model.config.hidden_size,
    f"hidden_size == {model.config.hidden_size}",
)
check(
    result.get("num_layers") == model.config.num_hidden_layers,
    f"num_layers == {model.config.num_hidden_layers}",
)
norm = result.get("final_norm")
check(
    isinstance(norm, float) and norm > 0.0 and norm == norm,  # norm == norm rules out NaN
    "final_norm is a positive, finite float",
)
print("Lesson 01: all checks passed.")
