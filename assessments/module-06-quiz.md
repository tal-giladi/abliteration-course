# Module 06 quiz - Running the real thing

Seven questions. Answers and explanations at the bottom - try all seven first.

---

**1.** A colleague runs `heretic --model Qwen/Qwen3-0.6B` in a CI pipeline and the job hangs
forever with no output going anywhere useful. What's actually happening, and which flags
would you add to make the exact same run finish unattended?

**2.** `heretic --help` prints almost instantly - no multi-second delay for loading `torch`,
`transformers`, or `optuna`. A real run of `heretic Qwen/Qwen3-0.6B` takes much longer just to
get going. Why does `--help` skip that cost?

**3.** `config.default.toml` sets `n_startup_trials = 60` and `n_trials = 200`, but this
module's exercise overrides `--n-trials 8`. What actually happens to the parameter search in
that case - is it still using TPE the way Lesson 15 described, or something else?

**4.** In `model.py::abliterate()`, the real weight matrix is upcast to `torch.float32` before
`lora_A`/`lora_B` are computed, even when the model itself is loaded in `bfloat16`. Why does
this matter, and why did this course's own `lib/ablate.py` never need to do it?

**5.** `get_abliterable_components()` and `get_layer_modules()` scan every layer with a long
list of `with suppress(Exception): try_add(...)` calls, instead of just hardcoding
`self_attn.o_proj` and `mlp.down_proj` the way this course's `lib/ablate.py` does. What is that
extra machinery actually for?

**6.** You add one new field, `my_flag: bool = Field(default=False, description="...")`, to the
`Settings` class in `heretic`'s `config.py`. List the three places that change becomes visible
without writing any additional code, and explain why all three come from one change.

**7.** You've just written a new scorer plugin and want to know it's correctly implemented
*before* committing to a real, multi-trial optimization run against a full-size model. What's
the fastest real verification you can do, and why doesn't it require loading a model at all?

---

## Answers

**1.** `heretic` is an interactive program: once optimization finishes, it prints a Pareto-front
selection menu and waits for arrow-key input, then asks what to do with the model (save /
upload / chat / benchmark), and a "save" answer asks for a directory - none of which work over
a plain CI pipe. Every one of those prompts is backed by a `Settings` field that defaults to
`None` ("ask"); setting it via a flag skips the prompt because `ask_if_unset` just returns the
value without calling `.ask()`. The minimum flags: `--trial-index <n>` (skip the Pareto-front
menu), `--model-action save --save-directory <path>` (skip the action menu and the path
prompt), and `--export-strategy merge` or `adapter` (skip the export-strategy prompt).

**2.** `main.py` checks `sys.argv` for `-h`/`--help` at the very top of the file, before any of
the heavy imports (`torch`, `transformers`, `optuna`, `lm_eval`, `peft`, `bitsandbytes`) run,
and if it's present, calls `Settings()` immediately - which triggers `pydantic-settings`'
`CliSettingsSource` to print the generated help text and exit, all using only the lightweight
`pydantic`/`pydantic-settings` machinery already imported. The heavy ML imports that follow in
the file are simply never reached.

**3.** Every trial still goes through Optuna's `TPESampler`, configured with
`n_startup_trials=60`. Since `8 < 60`, every one of those 8 trials falls in the sampler's
random-sampling phase - the same uniform-random behavior Lesson 14's `random_search` used - and
TPE's model-based sampling (which only kicks in after `n_startup_trials` trials have completed)
never actually activates. It's still a legitimate run: the multi-objective Pareto-front
*selection* logic (`study.best_trials`, sorted by score) runs exactly the same regardless of
how the trials were sampled, so `--trial-index 0` still picks the best trial out of the 8 that
were actually tried. It's a smaller, more random search, not a broken one.

**4.** `bfloat16` has roughly 7-8 bits of mantissa precision. Computing a dot product and a
rank-1 outer product in that precision, then writing the result straight into a `bfloat16`
weight, compounds rounding error across every layer in a way that's invisible on any single
number but measurably degrades the model. Upcasting to `float32` for the arithmetic and
downcasting only the final result back to the adapter's dtype is the standard fix. This
course's `lib/ablate.py` never needed it because `lib/common.py` hardcodes
`DTYPE = torch.float32` for every tensor from Lesson 01 onward - there was never a
lower-precision weight to upcast in the first place.

**5.** `heretic` supports a wide range of model architectures - standard dense transformers,
several MoE variants (with expert lists at different attribute paths), and hybrid models like
Qwen3.5 that mix standard attention layers with linear-attention (`GatedDeltaNet`) layers using
different module names. `try_add` inside a `suppress(Exception)` block lets the code
*attempt* each known attribute path per layer and silently do nothing when a given model
doesn't have it, so the same function works across every supported architecture instead of
assuming the one fixed shape this course's `Qwen3-0.6B` target happens to have.

**6.** `config.py`'s `Settings` class is a single `pydantic-settings` `BaseSettings`, and
`settings_customise_sources` layers a `CliSettingsSource` (with `cli_kebab_case=True`), an
`EnvSettingsSource` (prefix `HERETIC_`), and a `TomlConfigSettingsSource` on top of the exact
same field definitions. Adding one field therefore simultaneously becomes: (1) a
`config.toml` key (`my_flag = true`), (2) a CLI flag (`--my-flag`), and (3) an environment
variable (`HERETIC_MY_FLAG`) - all three read from the one place the field is actually defined,
with no extra plumbing written for any of them.

**7.** Call the class's own `validate_contract()` classmethod (inherited from `Plugin`/`Scorer`)
and confirm it subclasses `heretic.scorer.Scorer` and defines `get_score`. `heretic` itself
runs this exact check on every configured scorer at startup, before loading any model or
running a single trial (`Evaluator._load_and_init_scorers`), specifically so a broken plugin
(e.g. one that defines its own `__init__`, which the contract forbids) fails fast instead of
after an expensive model load. Running that same check yourself catches the same class of
mistake in under a second.
