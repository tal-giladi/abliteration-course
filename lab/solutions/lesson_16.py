import re
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = LAB_DIR / "heretic-run.log"
SAVE_DIR = LAB_DIR / "heretic-out"

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
    venv_bin = Path(sys.executable).parent
    for name in ("heretic.exe", "heretic"):
        candidate = venv_bin / name
        if candidate.exists():
            return str(candidate)
    return "heretic"  # fall back to PATH lookup


def run_heretic(log_path: Path = LOG_PATH) -> str:
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")

    result = subprocess.run(
        [_heretic_executable(), *HERETIC_ARGS],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text, encoding="utf-8")
    return log_text


def parse_refusal_and_kl(log_text: str) -> dict:
    trial_numbers = [int(n) for n in re.findall(r"Running trial (\d+) of \d+\.\.\.", log_text)]
    refusal_pairs = re.findall(r"(?<!Baseline )Refusals:\s*(\d+)/(\d+)", log_text)
    kl_values = re.findall(r"(?<!Baseline )KL divergence:\s*([0-9.]+)", log_text)

    selected_match = re.search(r"Restoring model from trial (\d+)\.\.\.", log_text)
    if selected_match is None:
        raise ValueError("Could not find a 'Restoring model from trial N...' line in the log.")
    selected = int(selected_match.group(1))
    idx = trial_numbers.index(selected)

    baseline_refusals_match = re.search(r"Baseline Refusals:\s*(\d+)/(\d+)", log_text)
    baseline_kl_match = re.search(r"Baseline KL divergence:\s*([0-9.]+)", log_text)
    if baseline_refusals_match is None or baseline_kl_match is None:
        raise ValueError("Could not find baseline score lines in the log.")

    num, den = refusal_pairs[idx]
    b_num, b_den = baseline_refusals_match.groups()

    return {
        "baseline_refusals": int(b_num) / int(b_den),
        "final_refusals": int(num) / int(den),
        "baseline_kl": float(baseline_kl_match.group(1)),
        "final_kl": float(kl_values[idx]),
    }
