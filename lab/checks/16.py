from _lib import check, fail

from exercises.lesson_16 import LOG_PATH, parse_refusal_and_kl, run_heretic

try:
    log_text = run_heretic()
except NotImplementedError:
    fail("run_heretic() is still a stub — fill in the TODOs in exercises/lesson_16.py")
except FileNotFoundError:
    fail(
        "`heretic` isn't installed (or isn't next to this venv's Python) — "
        "run `lab/.venv/bin/pip install heretic-llm` "
        "(or `lab/.venv/Scripts/pip.exe install heretic-llm` on Windows) first, "
        "per Lesson 16's 'Do this' step 1."
    )
except Exception as error:  # subprocess/timeout/etc. — surface a clear cause, don't crash raw
    fail(f"running heretic failed: {error}. Delete {LOG_PATH} to force a fresh attempt.")

check(bool(log_text.strip()), "heretic run produced output")
check(
    "Optimization finished" in log_text or "Restoring model from trial" in log_text,
    "the run reached trial selection (not just an early crash)",
)

try:
    result = parse_refusal_and_kl(log_text)
except NotImplementedError:
    fail("parse_refusal_and_kl() is still a stub — fill in the TODO in exercises/lesson_16.py")
except Exception as error:
    fail(
        f"could not parse the expected lines out of the log ({error}). "
        f"If a previous run failed partway through, delete {LOG_PATH} and re-run "
        "`bash lab/lab.sh check 16`."
    )

for key in ("baseline_refusals", "final_refusals", "baseline_kl", "final_kl"):
    check(
        key in result and isinstance(result[key], float) and result[key] == result[key],  # not NaN
        f"result has a numeric {key!r}",
    )

check(
    0.0 <= result["baseline_refusals"] <= 1.0 and 0.0 <= result["final_refusals"] <= 1.0,
    "refusal rates are valid fractions between 0.0 and 1.0",
)
check(
    result["final_refusals"] <= result["baseline_refusals"],
    "refusal rate did not increase: baseline "
    f"{result['baseline_refusals']:.2f} -> selected trial {result['final_refusals']:.2f} "
    f"(delete {LOG_PATH} and re-run to force a fresh random draw if this fails)",
)

# Heretic itself defines no single fixed "coherence passes" cutoff — KL divergence is one of
# two co-optimized objectives, not a pass/fail gate. This course picks a generous cap so the
# check catches an obviously wrecked model (near-random output) without pretending to be an
# official Heretic threshold.
check(
    result["final_kl"] < 5.0,
    f"KL divergence from the original model stayed bounded: {result['final_kl']:.4f} < 5.0",
)

print("Lesson 16: all checks passed.")
