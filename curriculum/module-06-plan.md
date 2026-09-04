# Module 06 plan - Running the real thing

**Lessons 16-18. Prerequisite: Module 05. Produces: a real local `heretic` run, a line-by-line
reading of its actual ablation function, and one real, verified change to its source.**

## Objectives

1. Install, configure, and run the real `heretic` CLI end to end, locally and (revisited)
   in Colab, and interpret its printed output.
2. Read `heretic`'s actual `model.py::abliterate()` and connect every part of it to the toy
   version built across Modules 03-04.
3. Make one small, scoped, verified change to the real codebase.

## Lessons

### 16 - Installing and configuring Heretic
- **Concept**: `pip install heretic-llm`; `config.default.toml`'s structure - prompt sets,
  scoring weights, search budget, quantization; how this maps onto Colab's GPU-backed run
  against `Qwen/Qwen3-8B` from earlier in this conversation, and why the same command works
  unchanged on the course's own `Qwen/Qwen3-0.6B` on a CPU (slower, but no quantization
  needed at that size).
- **Example**: `heretic --help` walked through flag by flag, cross-referenced against
  `config.default.toml`.
- **Practice**: run `heretic Qwen/Qwen3-0.6B` locally to completion; the checker parses its
  saved output/report and asserts refusal rate dropped and coherence stayed above Heretic's
  own default threshold.
- **Summary**: everything from Modules 01-05 was building toward being able to read this one
  command's output and know exactly what it did and why, instead of trusting it blindly.

### 17 - Reading `model.py::abliterate()` for real
- **Concept**: the production version of Lessons 07-09 and 11-12 - dtype upcasts for
  precision, `torch.linalg`, the real `peft` `LoraConfig` at rank 1, `get_abliterable_components()`
  choosing which modules to touch, `AbliterationParameters` as the exact strength-curve shape
  from Lesson 12; reading unfamiliar production code by first finding your own toy code
  inside it.
- **Example**: `abliterate()` opened side by side with `lab/lib/ablate.py`, annotated
  function by function.
- **Practice**: a structured annotation exercise - map each of 8 real functions/parameters to
  the toy-course equivalent that does the same job; checked against a reference mapping.
- **Summary**: this is deliberately the hardest lesson in the course, and also the whole
  point - production code is dense not because the ideas are hard, but because performance
  and robustness add layers around ideas you already have.

### 18 - Making a real, scoped change
- **Concept**: picking a change small enough to verify in one sitting - add one scorer to
  `scorers/`, or one config default, or one CLI flag - reading just enough of `main.py`/
  `config.py`/`plugin.py` to know where it plugs in; verifying a change to someone else's
  codebase the same way you'd verify your own: run it, don't just read it.
- **Example**: adding a length-based coherence sanity check as a second scorer, end to end.
- **Practice**: implement one of three provided scoped-change options in a fork of
  `lab/vendor/heretic`; the checker runs `heretic --help` (for a new flag) or a config-loading
  smoke test (for a new default) and asserts the change is live.
- **Summary**: you can now read an issue on this repository, or any transformer-tooling
  repository built the same way, and have a real shot at fixing it.

## Dependencies

16 -> 17 -> 18. Lesson 16 needs network access to install `heretic-llm` and download the
model (already cached from Module 01, reused here); Lesson 17-18 need `lab/vendor/heretic`
cloned in Lesson 15.

## Misconceptions to hit head-on

- "Production code uses fundamentally different math than the toy version." (It's the exact
  same projection and factorization from Modules 03-04, wrapped in dtype handling and error
  recovery - Lesson 17 is designed to make that recognition visible.)
- "I need to understand the whole repository before I can change anything." (Lesson 18 is
  scoped precisely so that's false - one function, one config key, one flag, verified by
  running it.)
- "The Colab run from earlier in this conversation and this lesson's local run are different
  things." (Identical command, identical library, different model size and hardware - the
  point of running it locally is seeing the same tool at a scale you can inspect end to end.)
