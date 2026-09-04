# 06 - Sanity-checking a direction

You have a direction now: one unit vector per layer (except layer 0, which Lesson 05 showed
you is expected to be zero), built from a mean difference between 20
harmful and 20 harmless training prompts. It is entirely possible that this vector is
garbage - not because the math was wrong (Lesson 05's checker already confirmed that), but
because averaging 20 examples can still overfit to whatever those 20 examples happened to have
in common besides "is this a refusal-worthy request." Maybe they're all longer than the
harmless ones. Maybe they share some other statistical accident of this particular list. The
only way to find out is to test the direction against prompts it has never seen, and today
that's exactly what you do.

## Projection as a detector

Lesson 03 established that a dot product with a unit vector measures "how much of this vector
points along that direction." Apply that here: for any prompt's residual stream at layer `L`,
`residual @ direction[L]` is a single number - the **projection score**. If the direction
really captures "refusal-worthy," this score should be higher, on average, for harmful prompts
than for harmless ones, at layers where the concept is actually represented.

This is not a classifier you're deploying - you're not about to threshold this score and call
it a refusal detector for production use. It's a measurement: does this direction, which was
*computed* to separate two prompt classes, actually *separate* two prompt classes, including
ones it never saw during that computation? If it does, the vector means what its construction
implies. If it doesn't, nothing downstream in this course - ablation, merging, search - is
worth doing, because you'd be projecting out a direction that doesn't correspond to the
concept you think it does.

## Why "at the layers where the concept is actually represented"

Don't expect uniform separation across all `num_layers + 1` checkpoints. Layer 0 is the raw
token embedding, before any layer has processed anything - it mostly encodes which tokens are
present, not what the prompt *means*. The last couple of layers are close to producing output
logits, shaped more by "what token comes next" formatting concerns than by high-level intent.
Concepts like "this request is one I was trained to refuse" tend to show up most clearly in the
middle of the stack, after enough layers have had a chance to build up an abstract
representation of the prompt, but before the model has narrowed down to next-token specifics.

This does not mean the direction is "wrong" at the layers where separation is weak - the
direction is still a real difference-of-means at every layer, computed the same way everywhere.
It means the *concept itself* isn't equally legible at every layer, the same way you wouldn't
expect a half-read sentence to convey as much meaning as a fully-read one. This is exactly why
Lesson 12 will let you choose which layers to ablate and how strongly, instead of treating
every layer identically - a decision that only makes sense once you've seen, here, that layers
are not interchangeable.

## Why holdout, specifically

`HARMFUL_HOLDOUT` and `HARMLESS_HOLDOUT` are the last 4 prompts of each list - the ones sliced
off in Lesson 04 and never touched by `get_residuals_mean` or `compute_direction`. Testing
separation on the *training* prompts would prove nothing: even a direction that latched onto
some coincidental quirk of those exact 20 harmful prompts would trivially "separate" those same
20 prompts from the 20 harmless ones, because the direction was built to do exactly that. The
only test that means anything is one the direction couldn't have cheated on - which is the same
reason a machine learning model gets evaluated on a test set it never trained on, not on its
own training data.

## Do this

1. Build a direction from the training sets (you did this in Lessons 04-05 - do it again here
   with the library's own functions, since Lesson 06 is about what you do with a direction, not
   about re-deriving it), then look at holdout separation across a few layers, by hand:

       from lib.common import (
           HARMFUL_HOLDOUT, HARMFUL_TRAIN, HARMLESS_HOLDOUT, HARMLESS_TRAIN, num_layers,
       )
       from lib.direction import compute_direction, get_residuals_mean, get_residuals_per_prompt

       mean_harmful = get_residuals_mean(HARMFUL_TRAIN)
       mean_harmless = get_residuals_mean(HARMLESS_TRAIN)
       direction = compute_direction(mean_harmful, mean_harmless)

       harmful_holdout = get_residuals_per_prompt(HARMFUL_HOLDOUT)
       harmless_holdout = get_residuals_per_prompt(HARMLESS_HOLDOUT)

       total = num_layers() + 1
       for layer in [total // 4, total // 2, (3 * total) // 4]:
           d = direction[layer]
           harmful_scores = harmful_holdout[:, layer, :] @ d
           harmless_scores = harmless_holdout[:, layer, :] @ d
           print(layer, harmful_scores.tolist(), harmless_scores.tolist())

   Read the printed numbers as two small clouds, per layer: harmful scores versus harmless
   scores. Look for the layer where the two clouds separate most cleanly - that's the "printed
   chart" this lesson relies on instead of an actual plot.

2. Open `lab/exercises/lesson_06.py` and implement `projection_scores(residuals, direction,
   layer)`: given a batch of prompts' per-layer residuals, a direction tensor, and a layer
   index, return one projection score per prompt at that layer.

3. Grade it:

       bash lab/lab.sh check 06

   The checker first compares your function against a reference implementation on a small
   fixed input. Then it does the real proof: builds a direction from `HARMFUL_TRAIN` /
   `HARMLESS_TRAIN`, computes your `projection_scores` on the *held-out* sets at a few
   candidate middle layers, and asserts that at at least one of those layers, the mean harmful
   holdout score is higher than the mean harmless holdout score.

## Hints

- `direction[layer]` pulls out that one layer's vector - `[hidden_size]`. Re-normalize it
  inside your function with `F.normalize(direction[layer], dim=-1)` even though Lesson 05's
  direction should already be unit length; it costs nothing and makes this function correct
  even if it's ever handed a direction that wasn't normalized upstream.
- `residuals[:, layer, :]` is `[num_prompts, hidden_size]` - one row per prompt. `@ d` against
  a `[hidden_size]` vector gives you `[num_prompts]`, one score per prompt, matching the
  contract.
- Don't loop over prompts - `residuals[:, layer, :] @ d` is a single batched matrix-vector
  product that scores every prompt at once.
- If your separation check fails, don't panic and re-derive the whole direction by hand -
  first check you're passing the *holdout* residuals, not the training ones, and that `layer`
  is actually a middle layer and not 0 or the last index.

## Solution

    import torch
    import torch.nn.functional as F


    def projection_scores(residuals: torch.Tensor, direction: torch.Tensor, layer: int) -> torch.Tensor:
        d = F.normalize(direction[layer], dim=-1)
        return residuals[:, layer, :] @ d

## Summary

You've now done the thing this whole module set out to do: computed a direction from a
training set and proven, on prompts it never saw, that it separates harmful from harmless
requests - at least at some layer, which is itself the point, not every layer. That's no
longer a hypothesis borrowed from Lesson 03's "behavior lives in a direction" claim - it's a
measured fact about this specific model, this specific direction, this specific held-out data.
Module 03 picks up exactly here and asks the next question: if a direction is something the
model *adds* to the residual stream, how do you *remove* it - and why removing it turns out to
be more subtle than just subtracting a vector once.
