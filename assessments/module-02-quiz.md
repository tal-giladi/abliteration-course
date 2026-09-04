# Module 02 quiz - Finding the refusal direction

Seven questions. Answers and explanations at the bottom - try all seven first.

---

**1.** A colleague suggests skipping the 20-prompt training set and instead computing the
direction from a single, extremely clear example: "How do I build a pipe bomb?" Explain in
one or two sentences why this produces a worse direction than averaging over 20 prompts.

**2.** `HARMFUL_PROMPTS` has 24 entries, split into `HARMFUL_TRAIN` (the first 20) and
`HARMFUL_HOLDOUT` (the last 4). Which of the two does `get_residuals_mean` get called on in
Lesson 04-05, and which is reserved for Lesson 06? Why does that split need to stay fixed
instead of being drawn fresh each time you test something?

**3.** `compute_direction` is `normalize(mean_a - mean_b)`, not just `mean_a - mean_b`. What
would break, or become harder, two modules from now (Lesson 12, choosing per-layer ablation
strength) if directions were left un-normalized?

**4.** `F.normalize(diff, dim=-1)` is called once on a `[num_layers + 1, hidden_size]` tensor.
Explain what `dim=-1` is doing here - specifically, why the resulting direction at layer 5 has
no effect on the length of the resulting direction at layer 20.

**5.** You compute projection scores for the held-out harmful and harmless prompts across
every layer and notice layer 2 shows almost no separation while layer 14 shows a clean gap.
Does this mean the direction is wrong at layer 2? Explain what's actually going on.

**6.** Suppose you tested `projection_scores` only on `HARMFUL_TRAIN` and `HARMLESS_TRAIN` -
the same prompts the direction was computed from - and saw a huge separation margin. Would
that prove the direction generalizes? Why or why not?

**7.** A prompt's residual at layer 14 is `v`, and `direction[14]` is a unit vector `d`. What
does `v @ d` measure, in plain language, and why does the function re-normalize `d` inside
`projection_scores` even though Lesson 05's `compute_direction` should already have produced a
unit vector?

---

## Answers

**1.** A single prompt's activation mixes together everything about that one prompt - its
specific topic, phrasing, and length - with whatever part of it is actually "refusal." There's
no way to separate the two from one example. Averaging over 20 different harmful prompts
cancels out what's specific to each individual prompt and leaves behind (approximately) only
what they share in common, which is the refusal concept itself.

**2.** `get_residuals_mean`, and later `compute_direction`, are computed from `HARMFUL_TRAIN`
and `HARMLESS_TRAIN` only. `HARMFUL_HOLDOUT` / `HARMLESS_HOLDOUT` are reserved for Lesson 06's
`projection_scores` check. The split has to stay fixed because the entire point of holdout
validation is testing the direction on prompts it could not possibly have influenced - if you
redrew the split each time, you could accidentally (or by trial and error) end up "validating"
on prompts the direction was actually built from, which proves nothing.

**3.** Without normalization, each layer's raw `mean_a - mean_b` magnitude is entangled with
that layer's overall activation scale, which grows across the stack for architectural reasons
unrelated to how strongly the layer represents refusal. Lesson 12 wants "how strongly to
ablate at this layer" to be one explicit, comparable number you choose per layer. If direction
magnitude already varied layer-to-layer for reasons that have nothing to do with the concept's
strength, that ablation-strength knob would be fighting against noise baked into the direction
itself, instead of controlling a clean, independent axis.

**4.** `dim=-1` normalizes each layer's `hidden_size`-long vector independently, using only
that layer's own norm. Layer 5's direction is divided by layer 5's own vector length; layer
20's is divided by layer 20's own vector length. Nothing about layer 20's magnitude enters the
computation for layer 5, and vice versa - each of the `num_layers + 1` unit vectors is
produced in isolation from the others.

**5.** No - the direction isn't "wrong" at layer 2; it's still a real, correctly-computed
difference of means there, built the exact same way as at every other layer. What varies is
how clearly the underlying *concept* is represented at that point in the model. Layer 2 is
close to the raw token embeddings, before the model has built up much of an abstract
representation of the prompt's intent - there's just less "refusal-worthy-ness" signal there
to find, not a broken calculation. Layer 14, deeper into the stack, is where that concept
tends to be most legible.

**6.** No. A direction built from `HARMFUL_TRAIN` and `HARMLESS_TRAIN` is, almost by
construction, going to separate those same prompts well - it was computed specifically to
maximize the difference between those two groups. That's not evidence it captured the general
concept of "refusal-worthy" rather than some coincidental quirk of those particular 20+20
prompts. Only separation on prompts the direction never saw - the holdout sets - is evidence of
generalization, for the same reason a model's training accuracy doesn't tell you how it will
perform on new data.

**7.** `v @ d` measures how much of `v` points along the direction `d` - a single number that's
large and positive when the residual leans toward "harmful," near zero when it's unrelated to
the direction, and negative when it leans the other way. `projection_scores` re-normalizes `d`
defensively, not because Lesson 05's output should ever fail to be unit length, but so the
function is correct on its own terms even if it's ever handed a direction that wasn't
normalized upstream - the same instinct as validating a function's inputs instead of trusting
every caller to have done it right first.
