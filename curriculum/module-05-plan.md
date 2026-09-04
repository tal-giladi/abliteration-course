# Module 05 plan - Making it automatic

**Lessons 13-15. Prerequisite: Module 04. Produces: `lab/lib/score.py`, `lab/lib/search.py`,
and a working (if small) version of Heretic's own optimization loop.**

## Objectives

1. Score a model on two axes at once - did it stop refusing, did it stay coherent - and
   explain why optimizing only the first produces a broken model.
2. Implement a search loop that finds good ablation parameters instead of guessing them.
3. Read Heretic's actual optimizer and map every part of it back to what was just built.

## Lessons

### 13 - Scoring a model
- **Concept**: `refusal_rate` - a keyword-based heuristic, same category of tool Heretic's
  `scorers/` use, and why "good enough, fast, and a bit crude" beats a slow classifier for a
  search loop that calls it hundreds of times; `coherence_score` - comparing the edited
  model's next-token distribution to the original's via KL divergence, so "it stopped
  refusing" is never scored without also asking "is it still the same model".
- **Example**: score the unmodified model (near-zero coherence loss, high refusal rate on
  the harmful set by definition) as a baseline.
- **Practice**: implement `is_refusal`, `refusal_rate`, and `coherence_score` in
  `lab/lib/score.py`; checked against fixed example responses and logits with known scores.
- **Summary**: two numbers, from here on always reported together - a model that maximizes
  one at the total expense of the other has failed, not succeeded.

### 14 - Search instead of guesswork
- **Concept**: the parameter space from Lesson 12 (strength curve shape) is small but not
  small enough to grid-search by hand in reasonable time; a random-search loop that samples
  parameters, merges, scores, and keeps the best; why this is a legitimate (if unsophisticated)
  optimizer - and exactly what it's missing.
- **Example**: run 15 random trials over 2 parameters and print the best refusal/coherence
  tradeoff found.
- **Practice**: implement `random_search` in `lab/lib/search.py`; wire it to an objective
  function combining Lesson 13's two scores; the checker asserts the search beats a fixed
  baseline configuration on the held-out prompt sets.
- **Summary**: you've now automated the exact loop Heretic runs - propose parameters, merge,
  score, repeat - just with the simplest possible proposal strategy.

### 15 - Reading Heretic's real optimizer
- **Concept**: what a Tree-structured Parzen Estimator does that uniform random sampling
  doesn't - it models which regions of parameter space scored well and biases future samples
  toward them, instead of sampling blind every time; where this lives in the real repository
  (`main.py`'s Optuna integration, `evaluator.py`'s scoring calls) - cloned locally for this
  lesson only.
- **Example**: `git clone` the real `heretic` repository into `lab/vendor/heretic` (ignored by
  git) and open `src/heretic/main.py` side by side with your own `search.py`.
- **Practice**: a guided-reading exercise - answer questions by citing exact function names
  and line ranges from the real source; graded by a checker that confirms the cited functions
  exist and do what the answer claims (structural, not vibes-based).
- **Summary**: your random search and Heretic's TPE search solve the identical problem -
  Heretic's is just smarter about where to look next, which is the entire difference between
  "automatic" and "fast".

## Dependencies

13 -> 14 -> 15. Lesson 14's objective function is Lesson 13's two scores combined; Lesson 15
requires network access once, to clone the real repository, and no code from it is imported
anywhere in `lab/lib`.

## Misconceptions to hit head-on

- "Refusal rate alone tells you if the ablation worked." (A model that agrees with literally
  everything has a refusal rate of 0% and is useless - Lesson 13 exists to prevent optimizing
  for that.)
- "TPE is a completely different algorithm from random search." (It's the same
  propose-evaluate-repeat loop with a smarter proposal step - Lesson 15 is written to make
  that continuity, not a discontinuity, the takeaway.)
- "Bigger search budgets always help." (Past enough trials, a smarter sampler matters more
  than more trials of a dumb one - part of why Heretic doesn't just run more random trials.)
