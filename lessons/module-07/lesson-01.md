# 19 - Evaluating a decensored model properly

Module 05 gave you two numbers, and insisted you never look at one without the other:
`refusal_rate` (did it stop refusing) and `coherence_score` (did it stay the same model
otherwise). That pair caught the obvious failure mode — a search loop that finds the
"best" refusal rate by making the model agree with everything, including gibberish, isn't
success. But there's a second failure mode those two numbers can miss, and it's the one
you're most likely to hit by accident once you start turning the strength knob past what
Lesson 12 called reasonable: a model that stops refusing not because the refusal direction
was cleanly removed, but because ablation was heavy enough to break something else on the
way. Today you add the third number that catches it, and write the one function that
produces all three together.

## What coherence_score doesn't see

`coherence_score` (Lesson 13) compares the edited model's next-token logits to the
original's, on the same prompt, at a single forward pass — teacher-forced, one step. That's
cheap and useful, but it's a narrow window: it tells you the model's *immediate* prediction
hasn't drifted much, not that a full, freely generated response holds together. A model can
look locally sane one token at a time and still degrade over a 40-token generation — repeat
itself, trail off into unrelated tokens, lose the thread of what it was asked. Heavy-handed
ablation does exactly this: strip a strong enough direction out of every layer's output
projections and you're not surgically removing "refusal," you're perturbing a general-purpose
matrix that also does a hundred other things. The model stops refusing because it's
stopped being a particularly *reliable* model, full stop — and a single-step KL comparison
won't always catch that, because the damage compounds over generated tokens in a way one
forward pass can't see.

This is exactly the ambiguity a raw refusal-rate number can't resolve on its own: a 0%
refusal rate is consistent with "the ablation worked" and with "the model is too broken to
refuse anything, including things it should just be answering normally." Both look identical
on that one axis.

## A third axis: does it still know anything

The fix isn't a better coherence metric — it's a completely different kind of check,
answering a completely different question: not "does this look like the base model's
distribution," but "does this model still produce *correct* answers to things it should
trivially get right." `lib/score.py` already has this, as `CAPABILITY_QUESTIONS` — five
fixed general-knowledge questions with a short list of acceptable substrings each ("What is
the capital of France?" → `"paris"`) — and `capability_spot_check`, which runs a model's
answers against that list and returns the fraction it got right.

Be precise about what this is and isn't. **It is not a benchmark.** Five questions is not a
capability evaluation in any rigorous sense — it can't tell you the model got smarter,
dumber, or better at anything in particular. What it *can* tell you, cheaply, is whether the
model still knows that water is H₂O and a week has seven days. If it doesn't, whatever
happened during ablation broke more than refusal, no matter what the refusal rate says. A
model that fails this spot-check has not been successfully decensored — it's been damaged,
and a low refusal rate on a damaged model isn't a result worth keeping.

**Example.** Run the spot-check against the unmodified model and it should score at or near
5/5 — Qwen3-0.6B has no trouble with these on a good day. Run it against a model you
deliberately over-ablated (if you still have Lesson 12's high-strength experiment sitting
in a Python session) and watch the score drop, sometimes sharply, well before the refusal
rate would tell you anything's wrong. That gap — refusal down, coherence not obviously
terrible, capability visibly worse — is the exact failure mode this lesson exists to catch.

## Assembling the report

Refusal rate, coherence, capability — from here on, every evaluation this course does
reports all three together, and `lib/score.py`'s `full_report` just bundles the three numbers
you hand it into one dict. The part that isn't already written for you is the orchestration:
a function that actually *produces* the three numbers by running a model, not just packages
them once you already have them. That's today's second piece, `run_full_report`, and it's
the function Lesson 20's capstone will build directly on.

One wrinkle worth calling out before you write it: coherence needs something to compare
against — a "before" set of logits from an unmodified copy of the model. But by the time
you're evaluating a model in this course, Lesson 11 already told you the truth about
`merge_into_model`: it edits weights *in place*. Once you've merged an ablation into the
model this process has loaded, the unablated version is gone unless you saved a copy of its
logits (or the model itself) before merging. `run_full_report` can't magic that copy back
into existence after the fact — so it takes it as an optional argument, `base_logits`,
computed by whoever calls it, before any ablation happens. If the caller doesn't have one to
pass, coherence is reported as `1.0` — a placeholder meaning "not measured this run," not
"the model is definitely identical to base." Lesson 20 is where you'll see the *right* place
to capture `base_logits` — before the merge, in the same script.

## Do this

1. In a Python shell, look at what the model actually says to the five capability questions,
   unmodified:

       from lib.common import generate
       from lib.score import CAPABILITY_QUESTIONS

       prompts = [q for q, _ in CAPABILITY_QUESTIONS]
       for prompt, response in zip(prompts, generate(prompts)):
           print(prompt, "->", response)

   Compare what you see against each question's acceptable-answers list in
   `lib/score.py` — that's exactly the substring match `capability_spot_check` performs.

2. Open `lab/exercises/lesson_19.py` and implement `capability_spot_check(responses,
   questions=CAPABILITY_QUESTIONS) -> float` yourself — a from-scratch reimplementation of
   the function you just used above, with the identical signature. Lowercase each response,
   check whether any acceptable answer is a substring of it, and return the hit fraction.

3. In the same file, implement `run_full_report(model=None, base_logits=None) -> dict`:
   - Default `model` to `lib.common.load_model()` if not given.
   - Generate responses to `lib.common.HARMFUL_HOLDOUT` and score `lib.score.refusal_rate`.
   - Generate responses to the five `CAPABILITY_QUESTIONS` prompts and score them with
     *your own* `capability_spot_check` above.
   - If `base_logits` was given, forward the same five capability prompts through `model`,
     take the last-token logits, and score `lib.score.coherence_score` against `base_logits`.
     If it wasn't given, use `1.0`.
   - Return `lib.score.full_report(refusal, coherence, capability)`.

4. Grade it:

       bash lab/lab.sh check 19

   The capability-check half is graded by exact comparison against `lib.score`'s reference
   implementation on a fixed list of example responses. The `run_full_report` half runs
   against the real model and is graded on structure — the right keys, each value a float
   between 0.0 and 1.0 — not on a specific number, since what a real model actually produces
   can't be pinned down while writing this course.

## Hints

- `capability_spot_check`'s loop is `zip(responses, questions)`, unpacking each question as
  `(_prompt, acceptable)` — you only need the acceptable-answers list, not the prompt text
  itself, since the prompt was already sent to the model to get `responses`.
- `"h2o" in "the answer is h2o."` — substring, not exact match, and always lowercase the
  response first; don't lowercase the acceptable-answers list, it's already lowercase in
  `lib/score.py`.
- `run_full_report`'s capability prompts are `[q for q, _ in CAPABILITY_QUESTIONS]` — you need
  that list twice, once for generation and once implicitly via `capability_spot_check`'s
  default argument.
- Don't try to compute `base_logits` yourself inside `run_full_report` — by design, that's the
  caller's job, done *before* any ablation. Lesson 20 shows exactly where.
- `model(**inputs).logits[:, -1, :]` is the last-token logits you need for the coherence
  branch — same slicing pattern as every residual-stream extraction so far, just on `.logits`
  instead of `.hidden_states`.

## Solution

    from lib import common, score


    def capability_spot_check(
        responses: list[str],
        questions: list[tuple[str, list[str]]] = score.CAPABILITY_QUESTIONS,
    ) -> float:
        if not responses:
            return 0.0
        hits = 0
        for response, (_prompt, acceptable) in zip(responses, questions):
            lowered = response.lower()
            if any(answer in lowered for answer in acceptable):
                hits += 1
        return hits / len(questions)


    def run_full_report(model=None, base_logits=None) -> dict:
        model = model or common.load_model()

        harmful_responses = common.generate(common.HARMFUL_HOLDOUT)
        refusal = score.refusal_rate(harmful_responses)

        capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
        capability_responses = common.generate(capability_prompts)
        capability = capability_spot_check(capability_responses)

        if base_logits is not None:
            inputs = common.encode_chat(capability_prompts)
            edited_logits = model(**inputs).logits[:, -1, :]
            coherence = score.coherence_score(edited_logits, base_logits)
        else:
            coherence = 1.0

        return score.full_report(refusal, coherence, capability)

## Summary

"Did it stop refusing" was never the only question this course was asking — Module 05 added
"did it stay coherent," and today added "does it still know anything." Three numbers, one
function, every time you evaluate an ablated model from here forward. Lesson 20 is where all
three finally get produced by a single script that also does the ablating — no shell session
required, and nothing borrowed from a model you haven't built yourself.
