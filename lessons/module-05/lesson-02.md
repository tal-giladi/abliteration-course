# 14 - Search instead of guesswork

Lesson 12 gave you four knobs per component - `max_weight`, `max_weight_position`, `min_weight`,
`min_weight_distance` - and you picked values for them by eyeballing a plot. Lesson 13 gave you
two numbers to judge the result by. This lesson connects them: instead of you picking parameters
and checking the score by hand, a loop picks parameters, merges, scores, and keeps whatever
worked best - repeated as many times as you're willing to wait for. That loop is a **search**,
and building the simplest possible version of one is this lesson's whole job.

## Why "search" and not "solve"

You might expect there's a formula - take the derivative of coherence with respect to
`max_weight`, set it to zero, done. There isn't, not a usable one. The relationship between
"these four numbers per component, on this model" and "the resulting refusal rate and coherence
score" runs through a full forward pass of a many-layered neural network; it has no closed form,
it's expensive to evaluate (every candidate requires re-merging the ablation and running the
model on a prompt set), and it isn't even smooth in any way you could exploit with gradients -
you already turned off gradient tracking back in Lesson 02, because none of this course trains
anything. When you can't solve for the best parameters directly, and evaluating a candidate is
the only way to know how good it is, what's left is: propose a candidate, evaluate it, keep the
best one you've seen, repeat. That's a **search loop**, and it's the same shape whether the
"propose" step is dead simple or extremely clever - which is exactly the continuity Lesson 15
is about.

## The parameter space is small, but not hand-searchable

Two parameters, each in some numeric range, isn't "big" as search spaces go. But "small" doesn't
mean "gridable by hand in reasonable time." Say you wanted to check 10 values per parameter -
a coarse grid by any standard - over even just 2 parameters, that's 100 combinations, each one
requiring a full merge-and-score cycle. Real Heretic tunes closer to a dozen numbers at once
across every ablatable component. Nobody is manually trying a hundred-plus combinations and
writing down the results in a spreadsheet. A loop that samples parameters and tracks the best one
it's found does in seconds what hand-tuning would take hours to approximate worse.

## Uniform random search, precisely

`random_search` takes four things: an `objective` function to maximize, a `param_space` dict
mapping each parameter name to a `(low, high)` range, how many `n_trials` to run, and a `seed`
for reproducibility. Each trial: draw one uniformly random value per parameter from its range,
call `objective(**params)` with those values as keyword arguments, and record the `(params,
score)` pair. After `n_trials` trials, return whichever `params` scored highest, that score
itself, and the full trial-by-trial history.

There's nothing hidden here - `random.Random(seed).uniform(low, high)` for each parameter, each
trial, `n_trials` times, keeping a running best. It doesn't reuse anything it learned from trial
3 when picking trial 4's parameters - every draw is independent and uniform over the whole range,
every time. That's what makes it "the simplest possible optimizer" and not a strawman: it's
missing exactly one idea (use past results to bias future guesses), and every other automatic
search technique you'll ever encounter - including the one Lesson 15 opens up for real - is a
variation on filling that one gap in.

## The objective: Lesson 13's two scores, combined into one

`random_search` maximizes a single `objective` function. Lesson 13 gave you two - refusal rate
(lower is better) and coherence score (higher is better). Turning two into one is a real design
decision, not a formality, and the direction this course takes is deliberately simple: something
in the shape of

    objective_value = (1 - refusal_rate) + coherence_score

which rewards low refusal and high coherence in the same units, and which a broken model (0%
refusal rate achieved by producing garbage, coherence collapsed toward 0) cannot maximize,
because the coherence term punishes it right back. This is Lesson 13's misconception, made
concrete inside the actual search loop this time: an objective built from refusal rate alone
would have `random_search` cheerfully converge on exactly the degenerate, over-ablated model
Lesson 13 warned you about. Combine the two scores, and the search can't get there without
paying a price on the axis it just broke.

In your own objective function - the one you wire up around `random_search` for the practice
exercise below - the shape of the combination is yours to choose (a straight sum, a weighted
sum favoring one axis, anything monotonic in the right direction on each score works), but it
must move in the correct direction on *both* axes, or the search will find whichever axis you
left exposed.

## What this loop is missing, named honestly

Every trial `random_search` runs is independent of every other trial - it never asks "trial 7
scored badly at these parameter values; region nearby is probably also bad; let's not waste trial
12 there." A human tuning by hand does exactly that instinctively - after a couple of bad guesses
near one edge of a range, you stop trying values near that edge. Uniform random sampling has no
such instinct built in; it will happily resample a bad neighborhood 15 trials later. That's the
entire, exact gap Lesson 15's TPE sampler fills - not a different problem, not a different loop
shape, one specific missing idea inside the identical propose-evaluate-keep-the-best structure.

## Do this

1. Open `lab/exercises/lesson_14.py` and implement `random_search`, matching the signature above:

       def random_search(
           objective: Callable[..., float],
           param_space: dict[str, tuple[float, float]],
           n_trials: int = 15,
           seed: int = 0,
       ):

   Return `(best_params, best_score, history)`.

2. Seed a `random.Random(seed)` once, outside the trial loop - re-seeding inside the loop would
   make every trial draw the same "random" values.

3. Run `n_trials` trials. Each one draws a fresh value per parameter from `param_space` (a plain
   dict comprehension over `param_space.items()` does this in one line), calls
   `objective(**params)`, and appends `(params, score)` to `history`.

4. Track whichever trial has produced the highest score seen so far as `best_params` /
   `best_score`, updating them whenever a new trial beats the current best (strictly beats - see
   the hints for why `>` and not `>=` matters here).

5. Grade it:

       bash lab/lab.sh check 14

   The checker does **not** compare your output number-for-number against a reference run - your
   random draws won't necessarily land in the same order as anyone else's, and that's fine by
   design. Instead it defines a fixed, simple 2D objective with a known best answer, runs *your*
   `random_search` against it with a fixed seed and 30 trials, and checks that you found a point
   close to the true optimum, that `history` really has 30 entries, and that `best_score` really
   is the best score in `history` - i.e. it grades what a search loop is supposed to *do*, not
   whether your random numbers matched someone else's.

## Hints

- Build `params` with a dict comprehension over `param_space.items()`:
  `{name: rng.uniform(lo, hi) for name, (lo, hi) in param_space.items()}` - one call to
  `rng.uniform` per parameter, per trial.
- `objective(**params)` - the dict's keys must match the objective function's keyword argument
  names exactly, which is why `param_space`'s keys are meaningful strings like `"x"` or
  `"strength"`, not arbitrary indices.
- Initialize `best_score = float("-inf")` before the loop, so the very first trial always
  becomes the initial best, regardless of what it scores.
- Use strict `>` when updating the best, not `>=` - ties should keep the *earliest* best trial,
  which is both a reasonable convention and what makes "the best entry in `history`" unambiguous
  when two trials tie exactly.
- `history` should be a flat list of `(params, score)` tuples, one per trial, in the order the
  trials ran - not a dict, not grouped by parameter.

## Solution

    import random
    from typing import Callable

    def random_search(
        objective: Callable[..., float],
        param_space: dict[str, tuple[float, float]],
        n_trials: int = 15,
        seed: int = 0,
    ):
        rng = random.Random(seed)
        best_params, best_score = None, float("-inf")
        history = []
        for _ in range(n_trials):
            params = {name: rng.uniform(lo, hi) for name, (lo, hi) in param_space.items()}
            score = objective(**params)
            history.append((params, score))
            if score > best_score:
                best_params, best_score = params, score
        return best_params, best_score, history

## Summary

You've now built the exact loop Heretic runs when you point it at a real model: propose
parameters, apply them, score the result, keep the best, repeat - just with the simplest possible
proposal strategy, one that samples blind every single time. Lesson 15 opens the real repository
and shows you what changes - and, just as importantly, what doesn't - when "sample blind" is
replaced with "sample using what every previous trial already told you."
