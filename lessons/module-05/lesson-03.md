# 15 - Reading Heretic's real optimizer

Lesson 14's `random_search` and Heretic's real optimizer solve the identical problem: propose
parameters, apply them, score the result, keep the best, repeat. This lesson doesn't teach you a
new algorithm from a diagram - it has you clone the actual `heretic` repository and go find the
places where that loop lives in real, running, production code. Reading unfamiliar code you
didn't write, in a repository you didn't design, is its own skill, and everything you've built in
Modules 01-05 is what makes this repository *readable* instead of intimidating: you already know
what a residual direction is, what ablation weight does, what refusal rate and coherence measure.
What's left is recognizing them under real names.

## What a Tree-structured Parzen Estimator does that uniform sampling doesn't

Heretic's sampler is **TPE** - a Tree-structured Parzen Estimator, and it comes from a library
called **Optuna**, a general-purpose hyperparameter optimization framework. The one-sentence
version of what it adds over `random_search`: it remembers every trial it's run, and it uses that
history to bias where it samples next, instead of sampling blind every time.

Concretely, and still without a new equation you need to derive: TPE splits its past trials into
"good" ones (scored well) and "not as good" ones, using some quantile cutoff. It then builds two
probability distributions over the parameter space - one describing where the good trials tended
to land, one for the rest - and the next trial is drawn from a region that the "good" distribution
considers likely and the "not as good" distribution considers unlikely. In practice this means:
after a run finds that `max_weight` around 0.9 tends to work and `max_weight` above 1.3 tends to
wreck coherence, TPE stops wasting trials up near 1.3 and starts concentrating them near 0.9 - the
exact instinct a human hand-tuning parameters develops after a few bad guesses, and the exact
thing Lesson 14 named as `random_search`'s one missing idea. TPE is not a different loop. It's the
same propose-evaluate-keep-the-best loop from Lesson 14, with a smarter "propose" step - which is
also why nothing else in this course changes: the model, the scores, the four ablation
parameters per component, all identical. Only the sampling strategy differs.

## Cloning the real repository

This is the one lesson in the course that reaches outside `lab/lib` on purpose. Clone Heretic
into `lab/vendor/heretic` - a path this course's `.gitignore` already excludes, specifically for
this lesson, so cloning it won't pollute this repository:

    git clone https://github.com/p-e-w/heretic lab/vendor/heretic

Nothing from that clone is imported by any file under `lab/lib` or `lab/exercises` - this lesson's
exercise is a guided reading exercise, graded by checking that you actually looked at the right
files, not by running any of Heretic's code. You don't need a GPU, a downloaded model, or even a
working `torch` install to do this lesson - open the files in an editor and read.

Three files matter here: `src/heretic/main.py` (where the search itself runs), `src/heretic/
evaluator.py` (where a trial's parameters get turned into scores), and `src/heretic/scorers/`
(the actual refusal-rate and coherence equivalents, as separate, swappable plugins - Heretic
calls them "scorers").

## Where the study gets created

Open `src/heretic/main.py` and look inside the `run()` function. Past the CLI argument handling,
model loading, and prompt loading, you'll find:

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            seed=settings.seed,
        ),
        storage=storage,
        directions=directions,
        study_name="heretic",
        load_if_exists=True,
    )

That's the whole "use TPE instead of uniform random" decision, made in one place: `TPESampler`,
imported from `optuna.samplers`, handed to `optuna.create_study`. A few details worth noticing,
each one a real design decision and not an accident:

- `n_startup_trials` - TPE needs *some* trials on the board before "good region vs. bad region"
  means anything. Before that count is reached, it samples uniformly at random - literally
  running Lesson 14's algorithm until it has enough history to do better.
- `multivariate=True` - models the parameters jointly rather than one at a time, so it can learn
  "these two parameters work well *together*," not just "this one parameter, marginally, tends to
  work."
- `directions` (plural) - Heretic optimizes more than one objective at once (refusal rate and
  coherence, same as your own combined objective from Lesson 14, but kept as *separate* objectives
  here rather than summed into one number - Optuna supports genuine multi-objective optimization,
  returning a Pareto front of trials instead of one single "best," which is why the CLI later lets
  you pick among several finished trials rather than handing you just one).
- `load_if_exists=True` and the `storage` object above it - the study can be paused and resumed
  across runs, which is also why `study_checkpoint_file` and a `JournalStorage` show up just
  above this block. Nothing in Lesson 14's toy version needed this, since it runs to completion in
  one call.

## Where the objective function lives

Search back up from `optuna.create_study` a little further in the same `run()` function and
you'll find a nested function, literally named `objective`:

    def objective(trial: Trial) -> tuple[float, ...]:
        ...
        model.reset_model()
        model.abliterate(residual_directions, direction_index, parameters)
        scores = evaluator.get_scores()
        objective_values = evaluator.get_objective_values(scores)
        ...
        return objective_values

Read past the print statements and progress bookkeeping and the shape is exactly Lesson 14's loop
body, just with real pieces slotted in instead of a toy 2D function: `trial.suggest_float(...)`
and `trial.suggest_categorical(...)` calls (this is *how* Optuna's sampler actually proposes a
parameter - not `rng.uniform(lo, hi)` directly, but the trial object asking the sampler, which is
what lets the same `objective` function work identically whether the study is using `TPESampler`
or Optuna's own random sampler) stand in for your `param_space` draw, `model.reset_model()` +
`model.abliterate(...)` is `merge_into_model` from Module 04 wearing production clothes, and
`evaluator.get_scores()` is Lesson 13's `refusal_rate` / `coherence_score` call, generalized. The
function is called once per trial by `study.optimize(objective_wrapper, n_trials=...)`, further
down the same file - the direct equivalent of Lesson 14's `for _ in range(n_trials):` loop, just
driven by Optuna's `Study` object instead of a Python `for` statement.

Notice what `objective` returns: `objective_values`, a **tuple** of floats, not one combined
number. That tuple lines up positionally with the `directions` list from `create_study` above -
this is the mechanical difference between "optimize one blended score" (Lesson 14's approach, and
a legitimate one) and "optimize several scores at once and let the algorithm find trade-offs
between them" (Optuna's multi-objective mode, which is what Heretic actually uses).

## Where a trial actually gets scored

`evaluator.get_scores()` is a one-line call from inside `objective`, but it's worth opening
`src/heretic/evaluator.py` and finding it for real:

    def get_scores(self) -> list[tuple[str, Score]]:
        ctx = Context(settings=self.settings, model=self.model)
        return [
            (entry.name, entry.scorer.get_score(ctx)) for entry in self._scorer_entries
        ]

Every configured scorer - each one a small plugin class - gets asked for its `Score` on the
current (just-abliterated) model, through a shared `Context` object. `_objective_entries()`, just
below it in the same file, filters that list down to only the scorers actually contributing to
the optimization (a scorer can be computed and reported without being optimized against - Heretic
supports that too), and `get_objective_values()` turns that filtered list into the tuple `objective`
returns to Optuna.

## The real refusal-rate and coherence scorers

Open `src/heretic/scorers/keyword_rate.py`. The class is `KeywordRate`, and its `REFUSAL_MARKERS`
list is longer and messier than the one you typed into `lesson_13.py` - real-world phrasing is
messier than a course's fixed example set - but `_is_match` does exactly what your `is_refusal`
does: lowercase the response, check whether any marker is a substring. `get_score` runs it over a
whole prompt set and returns `match_count / len(self.prompts)` - `refusal_rate`, under a different
name, inside a class instead of a bare function, configured to minimize instead of maximize (a
config detail, not a different idea).

Now open `src/heretic/scorers/kl_divergence.py`. `KLDivergence.get_score` computes
`F.kl_div(logprobs, self._baseline_logprobs, reduction="batchmean", log_target=True)` - the same
`F.kl_div` call from `lesson_13.py`, same `log_target=True`, on the same kind of `log_softmax`d
logits. The one real difference: Heretic's `KLDivergence` scorer returns the raw KL value
directly, configured with `optimization = "minimize"` in its scorer config, where your
`coherence_score` wrapped the same quantity in `exp(-KL)` and treated bigger as better. Two
framings of the identical underlying number - `coherence_score` picked the shape that composes
naturally into Lesson 14's single blended objective; Heretic's scorer plugin didn't need to, since
Optuna's multi-objective mode keeps every scorer's value separate anyway.

## Do this

1. Clone the repository if you haven't yet:

       git clone https://github.com/p-e-w/heretic lab/vendor/heretic

2. Open, in an editor, side by side if you can manage it: `lab/vendor/heretic/src/heretic/
   main.py`, `evaluator.py`, `scorers/keyword_rate.py`, and `scorers/kl_divergence.py` - next to
   your own `lab/lib/search.py` and `lab/lib/score.py`.

3. Open `lab/exercises/lesson_15.py` and fill in `ANSWERS`, a dict mapping a handful of guided
   questions to citations in the exact form `"path/to/file.py::name"` - the file path relative to
   `lab/vendor/heretic/src/heretic/`, and the exact function or class name where the answer
   actually lives. Every citation in this lesson above (`main.py::run`'s `optuna.create_study`
   call, `main.py`'s `objective` function, `evaluator.py`'s `get_scores`, `KeywordRate`,
   `KLDivergence`) is a real answer to one of the questions you'll find in the stub - go verify
   each one against the actual cloned files rather than copying it from this page, since the
   checker checks the real files, not this lesson's prose.

4. Grade it:

       bash lab/lab.sh check 15

   The checker confirms each cited file actually exists under `lab/vendor/heretic/src/heretic/`
   and that the cited name actually appears in it - a real, structural check against the repository
   you cloned, not a guess-the-wording quiz. If you haven't cloned the repository yet, it will
   tell you so and exit cleanly instead of crashing.

## Hints

- If `lab/vendor/heretic` doesn't exist yet, the checker's first message will tell you exactly
  what to run - re-read it before assuming something else is broken.
- The citation format is `"file.py::name"`, always two colons together as the separator, always
  the file path relative to `src/heretic/` (so `main.py`, not `src/heretic/main.py`, and
  `scorers/keyword_rate.py`, not just `keyword_rate.py`).
- `optuna.create_study` and the `objective` function are both defined inside the same top-level
  `run()` function in `main.py` - there's no separate function named `create_study_for_heretic`
  or similar to hunt for; the real code keeps them local to `run()` because they close over
  `settings`, `model`, and `evaluator`, which is also worth noticing as a pattern.
- The scorer classes matter more than exact line numbers here - cite `KeywordRate` and
  `KLDivergence` themselves (the class definitions), not a helper method buried deeper inside
  them, unless a question specifically asks for the method.

## Solution

Answers will vary in exact question wording, but every citation below is verifiable against the
real, cloned repository as of this course's writing:

    ANSWERS = {
        "where_is_the_optuna_study_created": "main.py::run",
        "where_is_the_search_objective_function_defined": "main.py::objective",
        "where_does_a_trial_actually_get_scored": "evaluator.py::get_scores",
        "which_scorer_class_is_the_refusal_rate_equivalent": "scorers/keyword_rate.py::KeywordRate",
        "which_scorer_class_is_the_coherence_equivalent": "scorers/kl_divergence.py::KLDivergence",
    }

## Summary

Your `random_search` and Heretic's TPE-driven `study.optimize` solve the identical problem, with
the identical shape - propose, apply, score, keep the best, repeat - and everything you built in
Lessons 13 and 14 maps directly onto real, running code once you know where to look: a direction
computation you already recognize, a `merge_into_model` you already recognize wearing the name
`abliterate`, a `refusal_rate` and a `coherence_score` you already recognize wearing the names
`KeywordRate` and `KLDivergence`. The entire difference between "automatic" (what you built) and
"fast" (what Heretic ships) is one smarter proposal step - and Module 06 is where you stop reading
this repository from the outside and actually run it.
