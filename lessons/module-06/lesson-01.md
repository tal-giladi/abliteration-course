# 16 - Installing and configuring Heretic

Everything since Lesson 01 has been building one thing: the ability to open `heretic`'s real
source and recognize what it's doing. Before that, though, you get to run it - the actual
published tool, not a toy version of it, against the same model this whole course has been
using. This lesson installs it, walks the configuration surface for real, and gets you a real
before/after number out of a real run.

## Installing it

`heretic` ships on PyPI as `heretic-llm` (the import name is `heretic`; the package name had
to be different because `heretic` was already taken). Install it into this course's existing
venv - the one `lab/lab.sh up` already created and that already has `torch`/`transformers`
cached and working:

    lab/.venv/bin/pip install heretic-llm          # Linux/macOS
    lab/.venv/Scripts/pip.exe install heretic-llm  # Windows / Git Bash

This pulls in a real dependency stack - `optuna` (the search library Lesson 15 read), `peft`
(real LoRA adapters, not the plain tensors `lib/ablate.py` returns), `bitsandbytes`
(quantization), `lm_eval` (the benchmark harness), `questionary` (the interactive prompts
you're about to route around), and `rich` (console output). None of that is optional reading
for this course - Lesson 17 opens `model.py` and you'll see `peft` and `torch.linalg` used
directly.

One practical thing worth doing before your first run: `heretic` downloads models through the
regular Hugging Face cache, which defaults to `~/.cache/huggingface` - a *different* location
than this course's `lab/.cache` (set via `HF_HOME` in `lib/common.py`). Since you already have
`Qwen/Qwen3-0.6B` cached from Lesson 01, point `heretic` at the same cache instead of
downloading a second copy:

    export HF_HOME="$(pwd)/lab/.cache"   # run from the repo root, once per shell

## `config.default.toml`, for real

Module 05 had you build a scorer and a random-search loop by hand. `heretic`'s real
`config.default.toml` (in the package's source, one directory up from `src/heretic/`) is the
configuration surface for a system built on exactly those two ideas, plus a model-loading
layer neither of your toy modules needed. Walking it top to bottom:

- **`dtypes`** - a *list*, not a single value: `["auto", "float16", "bfloat16", "float32"]`.
  `heretic` tries each in order and falls back if one fails to load or produces broken output
  (a quick generation test catches `inf`/`nan` results). Your course's `lib/common.py` just
  hardcodes `DTYPE = torch.float32` because `Qwen3-0.6B` on a CPU never has this problem -
  production code can't make that assumption for an arbitrary model on arbitrary hardware.
- **`quantization`** - `"none"` or `"bnb_4bit"`. This is the flag from the Colab command this
  course keeps referencing: `heretic Qwen/Qwen3-8B --quantization bnb_4bit`. An 8B model
  doesn't fit comfortably in a free Colab GPU's VRAM at full precision; 4-bit quantization via
  `bitsandbytes` shrinks it enough to fit. `Qwen3-0.6B` needs none of that - at 600M parameters
  in float32 it's under 3GB, which is why this lesson's command has no `--quantization` flag
  at all. Same tool, same command shape, different model size, different flag - not a
  different tool.
- **`good_prompts` / `bad_prompts`** - the real prompt sets. By default, 400 prompts each from
  `mlabonne/harmless_alpaca` and `mlabonne/harmful_behaviors` on the Hugging Face Hub - the
  same two datasets your `HARMLESS_PROMPTS`/`HARMFUL_PROMPTS` in `lib/common.py` were a
  24-prompt, hand-picked stand-in for. These are used to compute the per-layer residual
  directions, exactly like Lesson 05's `compute_direction`.
- **`scorers`** - a list of plugin references, each with an `optimization` direction
  (`"minimize"`, `"maximize"`, or `"none"`). The default two are `KeywordRate` (counts refusal
  phrases in responses to `bad_prompts` - the production version of `lib/score.py`'s
  `is_refusal`/`refusal_rate`) and `KLDivergence` (KL divergence from the original model's
  next-token distribution on `good_prompts` - the production version of `coherence_score`,
  minimized instead of maximized, but measuring the same drift). Lesson 18 adds a third.
- **`n_trials`** and **`n_startup_trials`** - default `200` and `60`. This *is* Lesson 15's
  `TPESampler`: the first `n_startup_trials` trials sample randomly (exactly like your
  `random_search` from Lesson 14), then TPE starts using what it's learned to sample smarter.
- **`orthogonalize_direction`**, **`row_normalization`**, **`winsorization_quantile`** - real
  refinements to the projection math Module 03 derived, covered when Lesson 17 opens
  `abliterate()` itself. Leave them at their defaults for this run.

Every one of these is also a CLI flag (`--quantization bnb_4bit`, `--n-trials 8`, and so on) -
`config.py` builds `Settings` as a single `pydantic-settings` class that accepts a TOML file,
environment variables (prefixed `HERETIC_`), *and* the command line, all reading from the same
field definitions. Run `heretic --help` to see the full, real flag list generated from that
class - it's long, and every flag on it corresponds to something you just read in
`config.default.toml`.

## How a run actually ends

Here's the part the Colab run from earlier in this conversation glossed over, because a
notebook cell hides it: `heretic` is an *interactive* program. After the optimization loop
finishes, it prints a "Pareto optimal trials" menu and waits for you to pick one with arrow
keys; after that, it asks what to do with the model (save / upload / chat / benchmark / go
back); saving asks for a directory and an export strategy. None of that works over a plain
pipe or in a script - it needs a real terminal for the arrow-key menus.

The way out is exactly the same mechanism that makes every setting a CLI flag: every one of
those interactive questions is backed by a `Settings` field that starts out `None` (meaning
"ask"). Set the field - via a flag - and the prompt is skipped entirely, because
`ask_if_unset(value, question)` just returns `value` without ever calling `question.ask()`
when `value` isn't `None`. So a fully unattended run is just:

    heretic \
      --model Qwen/Qwen3-0.6B \
      --n-trials 8 \
      --checkpoint-action restart \
      --trial-index 0 \
      --model-action save \
      --export-strategy adapter \
      --save-directory lab/heretic-out

Reading this against the flags you now recognize from `config.default.toml`: `--n-trials 8`
overrides the default 200 (this course's CPU box doesn't need a multi-hour search to see the
mechanism work); `--checkpoint-action restart` skips the "resume previous run?" prompt if
you've run this before; `--trial-index 0` picks the best trial from the sorted Pareto front
without asking; `--model-action save` skips the action menu; `--export-strategy adapter` saves
just the rank-1 LoRA adapter (a few megabytes) instead of a full merged model copy of
`Qwen3-0.6B` (over a gigabyte) - the same reversibility argument from Lesson 09, applied to
disk space instead of just "can I undo this."

Run that command (from the repo root, with `lab/.venv`'s `bin`/`Scripts` on your `PATH`, or by
calling the venv's `heretic` executable directly) and redirect its output somewhere you can
read it back:

    heretic --model Qwen/Qwen3-0.6B --n-trials 8 --checkpoint-action restart \
      --trial-index 0 --model-action save --export-strategy adapter \
      --save-directory lab/heretic-out > lab/heretic-run.log 2>&1

On a laptop CPU this genuinely takes a while - each of the 8 trials generates real responses
to 100 evaluation prompts plus computes residual directions over 800 prompts total. Expect
somewhere from fifteen minutes to the better part of an hour, not seconds. That's the point of
this lesson: every earlier check in this course finished instantly because it ran a handful of
prompts through a function you wrote. This is what "the real thing" actually costs.

## What ends up in the log, and why you can parse it

`heretic` prints its progress with `rich`'s `Console.print`, using markup like `[bold]text[/]`.
When output goes to a real terminal, `rich` renders that as color. When it's redirected to a
file or a pipe (as it is above, and as it will be when your Python `subprocess` call captures
it in the exercise below), `rich` detects that and prints plain text instead - the markup tags
are still *parsed*, just not colorized. That's not a lucky accident you're relying on; it's
documented `rich` behavior, and it's exactly what makes grepping a captured CLI transcript for
real numbers a legitimate technique instead of a hack.

Two kinds of lines matter for this lesson's exercise. Once, near the top, before any trial
runs:

    * Baseline Refusals: 24/100
    * Baseline KL divergence: 0 (by definition)

And once per trial, inside `Running trial N of 8...`:

      * Refusals: 6/100
      * KL divergence: 0.1847

Right before your chosen trial is restored for saving, one more line names *which* trial
number that was:

    Restoring model from trial 5...

That's your real, verifiable signal: match the trial number in that line against the trial
numbers in the `Running trial N of ...` blocks, and you know exactly which `Refusals`/`KL
divergence` pair belongs to the model that actually got saved to `lab/heretic-out`.

## Do this

1. Install `heretic-llm` into `lab/.venv` and set `HF_HOME`, as shown above.
2. Open `lab/exercises/lesson_16.py` and implement `run_heretic()` (runs the command above via
   `subprocess`, caching the captured output to `lab/heretic-run.log` so repeated checks don't
   re-run a 20+ minute process) and `parse_refusal_and_kl(log_text)` (extracts the baseline and
   selected-trial `Refusals`/`KL divergence` numbers using the line shapes described above).
3. Grade it:

       bash lab/lab.sh check 16

   The first run actually invokes `heretic` end to end - budget real time for it. Every run
   after that reuses the cached log and returns immediately, until you delete
   `lab/heretic-run.log` to force a fresh one (useful if a single random trial happened to draw
   unlucky parameters).

## Hints

- `shutil.which` or a plain `Path.exists()` check tells you whether to reuse
  `lab/heretic-run.log` or actually run the command.
- Locate the `heretic` executable next to the venv's own Python
  (`Path(sys.executable).parent / "heretic"`, with a `.exe` fallback on Windows) rather than
  assuming it's on `PATH` - the check script runs under `lab/.venv`'s interpreter specifically.
- `subprocess.run(..., capture_output=True, text=True)` gives you combined access to
  `.stdout`/`.stderr` - `heretic` writes almost everything to stdout, but capture both and
  concatenate them so you don't miss anything if that ever changes.
- For a missing `heretic` executable, let `FileNotFoundError` propagate (or catch and re-raise
  with a clearer message) rather than swallowing it - the checker turns it into a clear
  "install it first" failure instead of a stack trace.
- `re.findall(r"Running trial (\d+) of \d+\.\.\.", log_text)` gives you the trial numbers in
  order; `re.findall(r"(?<!Baseline )Refusals:\s*(\d+)/(\d+)", log_text)` gives you each
  trial's refusal count/total in the same order (the negative lookbehind excludes the one
  baseline line, which also contains the word "Refusals:").

## Solution

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
        selected = int(selected_match.group(1))
        idx = trial_numbers.index(selected)

        baseline_refusals_match = re.search(r"Baseline Refusals:\s*(\d+)/(\d+)", log_text)
        baseline_kl_match = re.search(r"Baseline KL divergence:\s*([0-9.]+)", log_text)

        num, den = refusal_pairs[idx]
        b_num, b_den = baseline_refusals_match.groups()

        return {
            "baseline_refusals": int(b_num) / int(b_den),
            "final_refusals": int(num) / int(den),
            "baseline_kl": float(baseline_kl_match.group(1)),
            "final_kl": float(kl_values[idx]),
        }

## Summary

You just ran the actual published tool, end to end, unattended, and pulled a real before/after
number out of its own printed transcript instead of trusting a progress bar. Every flag you
used maps to a `Settings` field you can now find in `config.py`, and every number you parsed
came from a scorer you've already implemented a toy version of. Lesson 17 opens the file that
does the actual weight editing - `model.py::abliterate()` - and you'll find your own
`lib/ablate.py` inside it, buried under exactly the kind of production concerns
(`dtypes`, quantization, `peft`) this lesson just introduced.
