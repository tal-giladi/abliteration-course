# 18 - Making a real, scoped change

You can now read `heretic`'s core. This lesson uses that to make one small, real change to it -
and, just as importantly, to verify the change actually works by running something against it,
the same way you'd verify a change to your own code, not by reading it and hoping.

## Picking a change small enough to verify in one sitting

Three options, each touching a different, small part of the codebase you've now read pieces of.
Pick **one**.

**(a) A new scorer.** `config.default.toml`'s `scorers` list currently has two entries -
`KeywordRate` and `KLDivergence`, both classes in `src/heretic/scorers/`. A scorer is a class
that subclasses `heretic.scorer.Scorer`, implements one method, `get_score(self, ctx) -> Score`,
and returns a `Score(value, rich_display, md_display)` - `value` is what gets optimized (or just
reported, if you set `optimization = "none"`), `rich_display` is what you saw printed to the
console in Lesson 16, `md_display` is what ends up in a model card if the result is ever
uploaded. `KeywordRate` (`src/heretic/scorers/keyword_rate.py`) is a complete, short worked
example to copy the shape of. The module plan's suggested version: a length-based coherence
sanity check - responses that collapse to near-nothing, or balloon far past what a normal
answer looks like, are a cheap, fast signal that something broke, independent of what
`KLDivergence` measures.

**(b) A new config default.** Every field on the `Settings` class in `src/heretic/config.py`
becomes a line in `config.default.toml` - add a field, and there's a new documented default to
set.

**(c) A new CLI flag.** Here's the thing worth actually noticing: in this codebase, (b) and (c)
are the same change. `Settings` is a single `pydantic-settings` `BaseSettings` class, and
`settings_customise_sources` in `config.py` layers a `CliSettingsSource` (`cli_kebab_case=True`),
an `EnvSettingsSource` (prefix `HERETIC_`), and a `TomlConfigSettingsSource` on top of the exact
same field definitions. Add one field to `Settings`, and it simultaneously becomes a
`config.toml` key, a `--kebab-case-flag`, *and* an environment variable - with no extra code for
any of the three. That's not a coincidence you need to route around; it's the actual reason
adding a genuinely new *behavior* (option a) looks structurally different in the codebase from
adding a new *knob* (options b/c) - a knob is one field; a behavior is a whole new plugin class.

## Getting your change to actually run

You cloned `heretic`'s source into `lab/vendor/heretic` back in Lesson 15 (it's `.gitignore`d -
this course never commits someone else's AGPL-licensed source into its own repository). You
already have `heretic-llm`'s real dependencies installed from Lesson 16 - `torch`, `peft`,
`optuna`, and the rest. You do **not** need to reinstall anything to make your edits to the
clone take effect: Python resolves `import heretic` by walking `sys.path` in order, so putting
`lab/vendor/heretic/src` ahead of your installed `heretic-llm` package on `sys.path` makes every
`heretic.*` import resolve to *your edited clone* instead, while all its third-party
dependencies still come from the regular install. `lab/checks/18.py` does exactly this before
importing anything from `heretic` - you don't need to configure it yourself, but it's worth
understanding why the check works without a `pip install -e`.

- **If you picked (a):** create `lab/vendor/heretic/src/heretic/scorers/<your_name>.py`,
  following `keyword_rate.py`'s shape (a `Settings(BaseModel)` if you want configurable fields,
  a class subclassing `Scorer`, `get_score(self, ctx)` returning a `Score`). Optionally add it
  to the `scorers` list in `lab/vendor/heretic/config.default.toml` so a real run would actually
  use it - not required for this lesson's check, but required if you want to see it show up in
  a real `heretic` run's output.
- **If you picked (b)/(c):** add a field to the `Settings` class in
  `lab/vendor/heretic/src/heretic/config.py`, with a real `default=...` and a `description=...`
  (every other field has one - `heretic --help` uses it), and add the matching key to
  `lab/vendor/heretic/config.default.toml` with a short comment, matching the style already
  there.

## Verifying it, not just reading it

For (a), the fast, real check that doesn't require a multi-hour optimization run: import the
class and confirm it satisfies the plugin contract `heretic` itself enforces before ever
running it. `Scorer` inherits a real classmethod, `validate_contract()`, that `heretic` calls on
every configured scorer at startup (`Evaluator._load_and_init_scorers` in `evaluator.py`) - it
raises if your class defines its own `__init__` (plugins aren't supposed to; there's an `init(ctx)`
hook for setup instead). Running that same check yourself is the same verification `heretic`
would do, just without waiting for a model to load first.

For (b)/(c), the real check is `heretic --help` - run it (as a subprocess, so it's the actual
CLI parser doing the work, not a guess about what it would print) and confirm your new
`--kebab-case-flag` shows up in the output. This is the same flag-generation machinery Lesson
16 pointed you at; if your field is missing from the help text, something about the field
definition is wrong (a bad type annotation is the usual culprit), and you've found that out in
about a second instead of at the start of a real, unattended run.

## Do this

1. Confirm `lab/vendor/heretic` exists (clone it if you haven't: `git clone
   https://github.com/p-e-w/heretic lab/vendor/heretic`).
2. Pick one of the three options above and implement it inside `lab/vendor/heretic`.
3. Open `lab/exercises/lesson_18.py` and implement `describe_change()` - it doesn't contain
   your actual change (that lives in `lab/vendor/heretic`); it tells the checker which option
   you picked and where to find it.
4. Grade it:

       bash lab/lab.sh check 18

## Hints

- For option (a), `heretic.scorer.Scorer` and `Score` are importable as
  `from heretic.scorer import Scorer, Score` - match `keyword_rate.py`'s imports.
- For option (a), don't define `__init__` on your scorer class - use `init(self, ctx)` for any
  one-time setup (loading prompts, computing a baseline), exactly like `KeywordRate.init()` and
  `KLDivergence.init()` do. `validate_contract()` enforces this and the checker calls it for
  real.
- For option (b)/(c), a plain type (`bool`, `int`, `str`, `float`) with a `Field(default=...)`
  is enough - you don't need anything as elaborate as `DatasetSpecification` or `ScorerConfig`
  for a single new knob.
- `field_name.replace("_", "-")` is how a `Settings` field name becomes its CLI flag - match
  that exactly in `describe_change()`'s `field_name` so the checker can find the right
  `--flag` in `heretic --help`'s output.

## Solution

This solution implements option (a): a length-based coherence sanity check, as a second scorer
alongside `KLDivergence`.

`lab/vendor/heretic/src/heretic/scorers/response_length.py`:

    from pydantic import BaseModel, Field

    from heretic.config import DatasetSpecification
    from heretic.scorer import Context, Score, Scorer


    class Settings(BaseModel):
        prompts: DatasetSpecification = Field(
            default=DatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="test[:50]",
                column="text",
            ),
            description="Prompts to measure response length on.",
        )
        target_length: int = Field(
            default=60,
            description="Expected response length in tokens, for comparison.",
        )


    class ResponseLength(Scorer):
        """Flags responses that collapsed to near-nothing or ballooned far past a
        normal length — a cheap, fast coherence sanity check independent of KL
        divergence."""

        settings: Settings

        @property
        def reproducible(self) -> bool:
            return True

        @property
        def score_name(self) -> str:
            return "Response length ratio"

        def init(self, ctx: Context) -> None:
            self.prompts = ctx.load_prompts(self.settings.prompts)

        def get_score(self, ctx: Context) -> Score:
            responses = ctx.get_responses(self.prompts)
            lengths = [len(response.split()) for response in responses]
            avg_length = sum(lengths) / len(lengths) if lengths else 0.0
            ratio = avg_length / self.settings.target_length
            return Score(
                value=abs(1.0 - ratio),
                rich_display=f"[bold]{ratio:.2f}[/]x target",
                md_display=f"{ratio:.2f}x target",
            )

And in `lab/exercises/lesson_18.py`:

    def describe_change() -> dict:
        return {
            "option": "scorer",
            "module": "heretic.scorers.response_length",
            "class_name": "ResponseLength",
        }

## Summary

You've now installed the real tool, read its production ablation code line by line, and made
one verified change to it. That's the whole arc of this module: Lessons 01-15 built a version
of every idea small enough to run in seconds against a real model; Module 06 spent its three
lessons connecting that understanding to the actual codebase a research community uses today.
What's left is Module 07 - evaluating a decensored model properly, building the whole pipeline
from scratch as this course's capstone, and the part this module deliberately left out: where
this technique stops being research and becomes something you need a better reason for.
