# 20 - Capstone: mini-heretic.py, and where this stops being research

Nineteen lessons ago, a transformer was a black box you'd only ever called through an API.
Along the way you built, piece by piece, a small library that does the thing `heretic` does:
`lib/direction.py` finds a behavior's direction from real activations, `lib/ablate.py`
projects that direction out of a model's weights, `lib/score.py` measures whether it worked,
`lib/search.py` looks for good parameters instead of guessing them. Today you wire all four
together into one script, run it against a real model, and get back a real before/after
report — the exact shape Lesson 19 defined, produced this time by a pipeline you also built.
Nothing here calls into `lab/vendor/heretic`. That's not an arbitrary restriction — it's the
point. The capstone is where you find out whether the toy library actually works standalone,
or whether it only ever worked because it was leaning on the real tool somewhere you didn't
notice. It doesn't lean on anything. `lib/` is complete.

## What "mini-heretic" actually does, in order

Strip away the scoring and the report, and the pipeline is four steps, and you've already
built every one of them:

1. **Compute a direction** (Lesson 05). Run `HARMFUL_TRAIN` and `HARMLESS_TRAIN` through the
   model, take the mean last-token residual stream for each group at every layer, subtract,
   normalize. `lib.direction.get_residuals_mean` and `lib.direction.compute_direction` do
   exactly this, and you're calling the same functions Lesson 06 used to prove the result
   actually separates held-out prompts.
2. **Merge it into the model** (Lesson 11). `lib.ablate.merge_into_model` takes that
   direction and edits the attention output projection and MLP down-projection at every
   layer, in place, so the ablation is now a property of the weights, not a hook you have to
   remember to register.
3. **Generate and score** (Lessons 13 and 19). Run the harmful holdout set and the five
   capability questions through the now-ablated model, and score refusal, capability, and
   coherence.
4. **Report** (Lesson 19). `lib.score.full_report` bundles the three numbers.

The only genuinely new decision in this lesson is *where* to capture the "before" state
for coherence. Lesson 19 had to accept `base_logits` as an argument, because by the time you
call an evaluation function you might already be looking at an ablated model with no
unablated copy left. Here, you control the whole script from the top — which means you get
to put the capture in the right place yourself: **before step 2 runs**, on the model as
loaded, before a single weight has changed. Get that ordering backwards — capture logits
*after* the merge — and `coherence_score` will faithfully compare the ablated model to
itself, report something close to a perfect 1.0, and tell you nothing at all. The number
would be real. It just wouldn't mean what you think it means. This is exactly the kind of
mistake that runs, produces a plausible-looking float, and passes review — the only defense
is knowing, structurally, what "before" has to mean.

## Fixed strength, or a small search

Step 2 needs a `weights` list — one multiplier per layer, threaded through
`merge_into_model` since Lesson 08. You have two honest options, and this lesson leaves the
choice to you.

**A fixed, uniform strength** (e.g. `1.0` at every layer) is the simplest thing that could
possibly work, and it's what the solution below does. It's also blunt: Lesson 12 already
showed you that full-strength ablation on every layer tends to over-correct, because early
layers carry more than just the target concept. You'll very likely see refusal drop a lot —
possibly to zero — at some real cost to coherence and capability.

**A small `lib.search.random_search` sweep** over strength would very likely find a better
tradeoff, the same way it did in Lesson 14. The cost is real, though: `merge_into_model`
edits weights in place and doesn't undo itself, so each trial in a search needs to start from
a fresh, unablated copy of the model — which on this course's setup means reloading it from
the cache, once per trial, on top of the generation calls a search already needs to score
each trial. That's a lot more wall-clock time for a CPU model whose whole appeal was fast
iteration. Neither choice is wrong; a fixed strength is the pragmatic default for a script
you're about to run once and read the output of, and a search is the right call the moment
you actually care about the specific tradeoff you land on, not just proving the pipeline
works end to end.

## Do this

1. Open `lab/exercises/lesson_20.py` and implement `run_mini_heretic(model_name: str =
   "Qwen/Qwen3-0.6B") -> dict`:
   - If `model_name` isn't `lib.common.MODEL_ID`, raise a `ValueError` — this lab only has
     one model cached.
   - Load the model with `lib.common.load_model()`. If that raises `OSError` (nothing
     cached, no network), re-raise a `RuntimeError` telling the caller to run
     `bash lab/lab.sh up` first — don't let a bare transformers stack trace be the first
     thing someone sees.
   - Capture last-token logits on the five `lib.score.CAPABILITY_QUESTIONS` prompts, on the
     model exactly as loaded — this is `base_logits`, and it has to happen now.
   - Compute the direction from `lib.common.HARMFUL_TRAIN` / `HARMLESS_TRAIN` via
     `lib.direction`, and merge it into the model with `lib.ablate.merge_into_model`, using
     either a fixed strength or a small search (see above — either is fine, note which one
     you picked and why in a comment if you go with search).
   - Generate responses to `lib.common.HARMFUL_HOLDOUT` and score `refusal_rate`; generate
     responses to the five capability prompts and score `capability_spot_check`; take fresh
     logits on those same five prompts and score `coherence_score` against `base_logits`.
   - Return `lib.score.full_report(refusal, coherence, capability)`.

2. Grade it:

       bash lab/lab.sh check 20

   This one is genuinely slow — a real model load, real forward passes over the training
   sets to build the direction, an in-place merge across every layer, and two more rounds of
   live generation to build the report. That's expected for this course's small CPU model,
   not a sign something's wrong. The checker's bar is loose and honest: an unmodified,
   safety-tuned Qwen3-0.6B refuses at or near 100% of the harmful holdout set (you saw this
   yourself, back in Lesson 13), so the check only asserts `refusal_rate < 1.0` — that the
   merge changed *something* — plus that coherence and capability both landed in `[0.0,
   1.0]`. It does not assert a specific number, because what a real model actually produces
   on your machine isn't something this course can predict while being written.

## Hints

- `model_name`'s default should reference `lib.common.MODEL_ID`, not a hardcoded string — the
  same reasoning as Lesson 01's "don't hardcode numbers you can read off the model."
- Capture `base_logits` with the *same* `encode_chat` + `model(**inputs).logits[:, -1, :]`
  pattern Lesson 19 used for the coherence branch of `run_full_report` — you're doing the
  identical operation, just at a different point in the script.
- `common.num_layers(model) + 1` is the length `merge_into_model`'s `weights` list needs to
  be — one entry for the embedding table, one per decoder layer.
- Wrap the whole function in `@torch.no_grad()` — you're not training anything, and it's one
  decorator instead of four separate `with torch.no_grad():` blocks scattered through the
  function.
- If you're stuck on the OSError handling: you don't need to simulate it to write correct
  code for it. `try / except OSError as exc: raise RuntimeError(...) from exc` is enough —
  the checker exercises the success path, not the missing-cache path, on this machine.

## Solution

    import torch

    from lib import ablate, common, direction, score


    @torch.no_grad()
    def run_mini_heretic(model_name: str = common.MODEL_ID) -> dict:
        if model_name != common.MODEL_ID:
            raise ValueError(f"This lab only has {common.MODEL_ID} cached — pass the default model_name.")

        try:
            model = common.load_model()
        except OSError as exc:
            raise RuntimeError(f"{model_name} isn't cached. Run 'bash lab/lab.sh up' first.") from exc

        capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
        inputs = common.encode_chat(capability_prompts)
        base_logits = model(**inputs).logits[:, -1, :]

        mean_harmful = direction.get_residuals_mean(common.HARMFUL_TRAIN)
        mean_harmless = direction.get_residuals_mean(common.HARMLESS_TRAIN)
        directions = direction.compute_direction(mean_harmful, mean_harmless)

        weights = [1.0] * (common.num_layers(model) + 1)
        ablate.merge_into_model(model, directions, weights)

        harmful_responses = common.generate(common.HARMFUL_HOLDOUT)
        refusal = score.refusal_rate(harmful_responses)

        capability_responses = common.generate(capability_prompts)
        capability = score.capability_spot_check(capability_responses)

        inputs = common.encode_chat(capability_prompts)
        edited_logits = model(**inputs).logits[:, -1, :]
        coherence = score.coherence_score(edited_logits, base_logits)

        return score.full_report(refusal, coherence, capability)

## Where this stops being research

Everything above is a complete, working answer to "how does abliteration work" — you can
now build it from nothing, against a real model, and explain every line. That's not the same
question as "when should I run it," and this course has deliberately not dodged the second
one just because it's less comfortable than the first.

The legitimate uses are concrete, not hypothetical. **Safety research** — understanding how
and where a refusal behavior is represented is how the field figures out whether alignment
techniques are robust or just cosmetic, and that question doesn't get answered by refusing to
look. **Red-teaming your own deployment** — if you ship a model behind a safety layer, running
this technique against it, on infrastructure you control, tells you what that layer is
actually doing versus what you assumed it was doing, before someone with worse intentions
finds out for you. **Portability testing** — checking whether a behavior you're relying on
(or trying to remove for a legitimate downstream use, like a research sandbox) survives a
provider swap or a fine-tune, which is an engineering question this technique answers
directly.

There's a real line on the other side of that, worth being precise about, because "it's just
research" and "it's just a tool" both stop being true at the same point. Understanding *how*
refusal is represented, and removing it on a model you control to study what changes — that's
the first half of this course, and it's not the part in question. The line is crossed by what
you do with the output. Running `mini-heretic.py` against `Qwen/Qwen3-0.6B` and reading the
report is an experiment. Taking an ablated model and actually generating the content this
course's own `HARMFUL_PROMPTS` list describes — real phishing emails, real working
ransomware, real instructions for something on that list, produced to actually use them, not
to test whether the model *can* produce them — isn't an experiment anymore. It's the thing
the base model's safety training existed to prevent, done anyway, by someone who now
understands the mechanism too well to claim they didn't see the difference. Auditing whether
your own product's safety layer holds up against this technique is the first case. Using the
technique to get the answers that layer was built to withhold, for real-world use, is the
second. Same script, same three numbers in the report, a completely different act.

This course isn't going to draw that line for you in every specific case — it can't, and
pretending otherwise would be its own kind of dishonesty. What it can do, and has tried to do
throughout, is make sure you're never in a position to cross it without knowing you did. "Can
I do this" was answered the moment `refusal_rate` dropped in your terminal. "Should I" is a
different question, and this course expects you to keep asking it after the check passes,
not just before you decided to take the module.

## Summary

You built, by hand, across twenty lessons, the technique behind a real published tool —
found a behavior's direction from real activations, proved it generalized, derived why
projecting it out works, turned that into a permanent weight edit, learned to control its
strength, scored it on three axes instead of one, searched for good parameters instead of
guessing, and read the production version to see your own toy code looking back at you. Today
that became one script, run once, against a real model, printing its own report with nothing
borrowed. You can now open `heretic`'s actual source and recognize it. You can also state,
correctly, what this technique is for and what it isn't — and that second part was never
optional homework. It was always the other half of the material.
