# Lesson 16 - Installing and configuring Heretic
# Read lessons/module-06/lesson-01.md before filling this in.
#
# This exercise runs the REAL heretic CLI (installed via `pip install heretic-llm`
# in Lesson 16's "Do this" step 1) end to end against Qwen/Qwen3-0.6B, and parses
# its own printed transcript for a real before/after refusal-rate and
# coherence (KL divergence) number. Nothing here is mocked -- the numbers this
# produces are whatever the real search actually found.

import re
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = LAB_DIR / "heretic-run.log"
SAVE_DIR = LAB_DIR / "heretic-out"

# The exact flags from the lesson: small n_trials for a CPU-friendly run, and
# every interactive prompt pre-answered so the run completes unattended.
HERETIC_ARGS = [
    "--model", "Qwen/Qwen3-0.6B",
    "--n-trials", "8",
    "--checkpoint-action", "restart",
    "--trial-index", "0",
    "--model-action", "save",
    "--export-strategy", "adapter",
    "--save-directory", str(SAVE_DIR),
]


def _heretic_executable() -> str:
    """Locate the `heretic` console script installed alongside this venv's
    Python, rather than assuming it's on PATH."""
    # TODO: check `Path(sys.executable).parent / "heretic.exe"` and
    # `.../ "heretic"`; return whichever exists. Fall back to the bare string
    # "heretic" (PATH lookup) if neither does.
    raise NotImplementedError


def run_heretic(log_path: Path = LOG_PATH) -> str:
    """Run the real `heretic` CLI to completion and return its captured
    stdout+stderr, as a single string.

    Reuses `log_path` instead of re-running if it already exists -- a real
    run takes real minutes, and re-running it on every `check` invocation
    would make this the only lesson in the course with a slow feedback loop.
    """
    # TODO:
    # 1. If log_path already exists, read and return its contents.
    # 2. Otherwise, run [_heretic_executable(), *HERETIC_ARGS] with
    #    subprocess.run(capture_output=True, text=True, timeout=3600).
    # 3. Concatenate stdout + stderr, write it to log_path, and return it.
    #    (Let FileNotFoundError propagate if `heretic` isn't installed --
    #    the checker turns that into a clear message.)
    raise NotImplementedError


def parse_refusal_and_kl(log_text: str) -> dict:
    """Parse the real signal `heretic` prints: the baseline Refusals/KL
    divergence scores, and the Refusals/KL divergence of the ONE trial that
    got selected (--trial-index 0) and saved.

    Returns {"baseline_refusals": float, "final_refusals": float,
             "baseline_kl": float, "final_kl": float} -- refusals expressed
    as a rate (matches / total), not a raw count.
    """
    # TODO: see the lesson for the exact line shapes to regex out:
    #   "* Baseline Refusals: N/M"
    #   "* Baseline KL divergence: 0 (by definition)"
    #   "Running trial <i> of <n>..." followed later, in the same trial's
    #     block, by "  * Refusals: N/M" and "  * KL divergence: X.XXXX"
    #   "Restoring model from trial <i>..." -- <i> tells you WHICH trial's
    #   printed numbers above are the ones that actually got saved.
    raise NotImplementedError
