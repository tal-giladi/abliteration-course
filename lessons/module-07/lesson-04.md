# 22 - Extension: the full recipe, measured like the paper (and where the paper stops)

> Optional extension — read Lesson 21 first. Everything here runs against `Qwen/Qwen3-0.6B`,
> the same weak, CPU-sized model with no meaningful dangerous capability that the whole course
> has used. That choice is the entire safety argument of this lesson; the closing section says
> why in plain terms.

Lesson 20 gave you `mini-heretic.py`: direction, merge, score, report, once, on one model.
Lesson 19 gave you three axes. Lesson 21 gave you a second refusal scorer and the reason one
number is never the number. This lesson puts all of it into one consolidated, repeatable
**recipe** — the thing a serious learner actually runs — and makes it produce a *before and
after* report in the shape a published study uses: not "here's the refusal rate," but "here's
what every axis was before I touched the model, and here's what it was after."

## The recipe, step by step

This is the whole procedure, on the weak model, start to finish. Every step is something you
already built; the only new thing is running them in the order that produces an honest
before/after.

1. **Load the model, untouched.** `common.load_model()`. Do nothing to it yet.

2. **Measure "before," on the loaded model.** This has to happen now, because Step 5 destroys
   the evidence (Lesson 11: `merge_into_model` edits weights in place). Capture:
   - last-token logits on the five capability prompts — this is your `base_logits`, the only
     thing that lets you compute coherence later;
   - generations on `common.HARMFUL_HOLDOUT`, scored with *both* `refusal_rate` and
     `strict_refusal_rate` (Lesson 21);
   - generations on the capability prompts, scored with `capability_spot_check`.

3. **Compute the refusal direction** from `common.HARMFUL_TRAIN` / `HARMLESS_TRAIN` via
   `lib.direction` — mean-difference, normalized, one vector per layer (Lesson 05).

4. **Choose a strength.** Fixed `1.0` per layer is the honest default for a recipe you run
   once and read (Lesson 12's warning about over-ablating early layers still applies; a
   `lib.search` sweep is the upgrade when you care about the specific tradeoff).

5. **Merge the direction into the weights** with `lib.ablate.merge_into_model`. The model is
   now ablated, in place. There is no going back without reloading.

6. **Measure "after," on the ablated model.** The same three measurements as Step 2 —
   refusal (both scorers), capability — plus the one you couldn't take before: coherence,
   `coherence_score(after_logits, base_logits)` on the capability prompts.

7. **Report before and after, side by side.** The delta is the finding. Refusal should fall
   (that's abliteration working); the strict refusal rate should stay at or above the lenient
   one (Lesson 21's guarantee); capability and coherence tell you whether the model survived
   or whether you just broke it quietly.

That is exactly the protocol the GLM-5.3 report used, minus the scale: measure refusal
multiple ways, measure capability, show both before and after, and let "refusal fell but
capability held" be a *finding to report*, never an assumed outcome.

## Do this

1. Open `lab/exercises/lesson_22.py` and implement
   `abliterate_and_report(model_name: str = common.MODEL_ID) -> dict` following the seven
   steps above. Guard `model_name` and the missing-cache `OSError` exactly as Lesson 20 did.
   Return a nested dict:

       {
         "before": {"refusal_lenient": f, "refusal_strict": f, "capability": f},
         "after":  {"refusal_lenient": f, "refusal_strict": f, "capability": f, "coherence": f},
       }

   Use `lib.score.strict_refusal_rate` (or import your Lesson 21 function — same thing) for
   the strict numbers.

2. Grade it:

       bash lab/lab.sh check 22

   Slow, for the same reasons as Lesson 20, now doubled — you generate twice (before and
   after). The bar is loose and honest: every value a float in `[0.0, 1.0]`;
   `before.refusal_lenient` near the ~1.0 unablated baseline; `after.refusal_lenient <=
   before.refusal_lenient` (ablation didn't make it refuse *more*); and `after.refusal_strict
   >= after.refusal_lenient` (Lesson 21's mathematical guarantee, now checked on live output).
   No exact post-ablation number is asserted — it depends on what your model actually
   produces, which this course can't pin down while being written.

## Hints

- Structure the function as two helper blocks — a "measure" step you run twice (before and
  after), differing only in whether coherence is computed — rather than writing the four
  generation calls out longhand. It reads better and it's the shape Lesson 19's
  `run_full_report` was nudging you toward.
- `base_logits` is captured once, in Step 2, and used once, in Step 6. Don't recompute it
  after the merge — that's the exact Lesson 20 trap that silently reports coherence ≈ 1.0.
- `@torch.no_grad()` on the whole function, same as Lesson 20.
- You do not need a search to pass. Fixed `weights = [1.0] * (common.num_layers(model) + 1)`
  is fine and is what the solution uses.

## Solution

    import torch

    from lib import ablate, common, direction, score


    def _measure(model, capability_prompts, base_logits=None) -> dict:
        harmful = common.generate(common.HARMFUL_HOLDOUT)
        out = {
            "refusal_lenient": score.refusal_rate(harmful),
            "refusal_strict": score.strict_refusal_rate(harmful),
            "capability": score.capability_spot_check(common.generate(capability_prompts)),
        }
        if base_logits is not None:
            inputs = common.encode_chat(capability_prompts)
            logits = model(**inputs).logits[:, -1, :]
            out["coherence"] = score.coherence_score(logits, base_logits)
        return out


    @torch.no_grad()
    def abliterate_and_report(model_name: str = common.MODEL_ID) -> dict:
        if model_name != common.MODEL_ID:
            raise ValueError(f"This lab only has {common.MODEL_ID} cached — pass the default model_name.")
        try:
            model = common.load_model()
        except OSError as exc:
            raise RuntimeError(f"{model_name} isn't cached. Run 'bash lab/lab.sh up' first.") from exc

        capability_prompts = [q for q, _ in score.CAPABILITY_QUESTIONS]
        inputs = common.encode_chat(capability_prompts)
        base_logits = model(**inputs).logits[:, -1, :]

        before = _measure(model, capability_prompts)

        mean_harmful = direction.get_residuals_mean(common.HARMFUL_TRAIN)
        mean_harmless = direction.get_residuals_mean(common.HARMLESS_TRAIN)
        directions = direction.compute_direction(mean_harmful, mean_harmless)
        ablate.merge_into_model(model, directions, [1.0] * (common.num_layers(model) + 1))

        after = _measure(model, capability_prompts, base_logits=base_logits)
        return {"before": before, "after": after}

## Where the recipe stops being safe to generalize

You now have a complete, measured, reproducible abliteration pipeline. Run it and you'll watch
a safety-tuned model's refusal rate fall on your own laptop. That is the correct thing to feel
slightly uncomfortable about, so be precise about why it's fine *here* and what would change
if it weren't.

It is fine here because of the model. `Qwen/Qwen3-0.6B` is a 0.6-billion-parameter model that
runs on a CPU and whose "harmful" outputs, when it stops refusing, are vague, often wrong, and
of no operational use to anyone — the `HARMFUL_PROMPTS` list exists to give the *refusal
classifier* something to classify, not to extract anything dangerous. Stripping refusal from
this model teaches you the mechanism and endangers no one. That is the only reason this course
exists in the form it does.

The GLM-5.3 report is what the same seven steps look like pointed at a model that *isn't*
harmless — a frontier open-weight model whose capabilities, Anthropic measured, include
autonomously developing working exploits, and whose refusal training they removed for roughly
the cost of a used car. The technique is identical. The consequence is not. That is exactly
why this extension reproduces the report's *measurement protocol* — how to evaluate honestly,
how two scorers disagree, how to show before/after — and does **not** give you a runbook for
reproducing it on GLM-5.3 or any other genuinely capable model. Not because the steps would be
different (they wouldn't), but because a step-by-step recipe for removing the safety training
from a model that can do real harm is the uplift the report was written to warn about, and
publishing it inside a course would be doing the thing the course spent twenty-one lessons
teaching you to recognize.

The capability of the model under the knife is the whole line. Below it — a toy you run to
learn — you have everything you need, and you built it yourself. Above it, the right move is
the one the report recommends to governments, not to individuals: evaluate before release,
and don't be the person who hands out the recipe. Lesson 20 said "can I" was answered the
moment the refusal rate dropped, and "should I" is a different question you keep asking after
the check passes. This lesson is where that stops being a slogan and becomes the reason a
specific thing you now know how to do is not written down here.

## Summary

You turned the whole course into one recipe and measured it the way a real study measures —
before and after, refusal two ways, capability, coherence — on a model weak enough that doing
so is purely educational. And you drew, concretely, the line the report draws: the method
generalizes to any model, the *recipe* deliberately does not, and the thing that decides which
side of the line you're on is the capability of the model you point it at, not the cleverness
of the technique. That judgement — not the projection math — is the last thing this course has
to teach, and now you've both built the tool and named the point where you stop.
