# Module 07 plan - Capstone and responsible use

**Lessons 19-20. Prerequisite: all prior modules. Produces: `mini-heretic.py`, built and run
end to end, and a properly evaluated ablated model.**

## Objectives

1. Evaluate a decensored model the way Module 05 insisted on, plus a capability spot-check,
   and produce a real before/after report.
2. Assemble everything built across the course into one standalone script, run it against a
   real model, and state - accurately - what this technique is for and where it isn't.

## Lessons

### 19 - Evaluating a decensored model properly
- **Concept**: refusal rate and coherence (Module 05) plus a third axis Module 05 never
  needed - a small, fixed capability spot-check (five general-knowledge questions with
  checkable answers) to catch the failure mode where a model stops refusing *because* it's
  stopped being a coherent model at all, not because it was successfully ablated; what
  "broke the model" looks like numerically versus what "worked" looks like.
- **Example**: run the spot-check against the unmodified model (near-perfect) and against a
  deliberately over-ablated model from Lesson 12's exercise (visibly worse).
- **Practice**: implement a `capability_spot_check` scorer and a `full_report(model)`
  function combining all three scores; checked against the Module 06 run's saved output.
- **Summary**: "did it stop refusing" was never the only question this course was asking -
  Lesson 20 is where all three numbers get produced by a single script, together.

### 20 - Capstone: mini-heretic.py, and where this stops being research
- **Concept**: assembling Modules 01-05's library into one script that takes a model name and
  the course's prompt sets and produces an ablated model plus Lesson 19's report - no calls
  into the real `heretic` package, only `lab/lib`; closing discussion, in plain terms: this
  technique's legitimate uses (safety research, red-teaming a deployment's own guardrails
  before an attacker does, testing model portability across providers) and where applying it
  crosses into helping the model do the specific harmful things its safety training existed
  to prevent - a line this course draws explicitly rather than leaving as an exercise.
- **Example**: the capstone script run once, live, against `Qwen/Qwen3-0.6B`, printing its
  own report.
- **Practice**: write `mini-heretic.py` in `lab/exercises/`; the checker runs it end to end
  and asserts the produced report clears the same bar Lesson 16's real `heretic` run did -
  refusal rate down, coherence and capability score within Heretic's own default thresholds.
- **Summary**: you built, by hand, the technique behind a real published tool, end to end,
  against a real model - and you can now explain exactly what it does, why each piece is
  there, and why "I can" and "I should" are different questions this course expects you to
  keep asking.

## Dependencies

19 -> 20. Lesson 20 imports from `lab/lib` exclusively (Modules 01-05's work), not from
`lab/vendor/heretic` (Module 06's clone) - the capstone proves the toy library actually
works standalone, not that it can call the real one.

## Misconceptions to hit head-on

- "A capability spot-check is the same thing as a benchmark." (It's five fixed questions, a
  smoke test for "did this break the model," not a rigorous capability evaluation - the
  lesson says so explicitly rather than overselling it.)
- "This course is over once the capstone passes." (Lesson 20's closing section is graded
  reading, not filler - the assessment quiz includes it.)
- "Since the tool is public and the model is small, none of this matters." (The technique is
  identical regardless of model size; a course that teaches the mechanism without ever
  naming its misuse potential has taught half of it.)
