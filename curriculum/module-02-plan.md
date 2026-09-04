# Module 02 plan - Finding the refusal direction

**Lessons 04-06. Prerequisite: Module 01. Produces: `lab/lib/direction.py`'s
`compute_direction`, and a direction that provably separates harmful from harmless prompts.**

## Objectives

1. Explain why the refusal direction is computed as a *difference of means* over many
   prompts, not read off a single activation.
2. Compute a real per-layer direction from the course's harmful/harmless prompt sets against
   `Qwen/Qwen3-0.6B`.
3. Prove - numerically, on prompts the direction was never computed from - that the direction
   actually separates the two behaviors.

## Lessons

### 04 - Harmful vs. harmless prompt sets
- **Concept**: why one prompt gives you a point and you need a *cloud* of points to find a
  direction; the course's fixed 24/24 harmful/harmless prompt sets and the held-out 4/4 split
  reserved for Lesson 06; how heretic's own default set in `config.default.toml` is built the
  same way, just bigger.
- **Example**: run both prompt sets through `get_residuals_per_prompt` from Module 01 and
  look at the mean norm per group.
- **Practice**: compute `get_residuals_mean` for the harmful-train and harmless-train sets
  and check the returned shape and that the two means are, in fact, different.
- **Summary**: a direction is only as good as the contrast it's computed from - garbage
  prompt sets in, a meaningless direction out.

### 05 - Computing the direction
- **Concept**: `direction = normalize(mean_harmful - mean_harmless)`, applied independently
  at every layer; why you normalize (so later "how strongly to ablate" is a separate,
  controllable knob, not baked into the direction's own magnitude - see Lesson 12).
- **Example**: compute the direction at one middle layer and print its cosine similarity to
  itself (1.0, a sanity check that normalization worked) and to a random vector (~0).
- **Practice**: implement `compute_direction` in `lab/lib/direction.py`; checked against a
  reference direction within floating-point tolerance.
- **Summary**: you now have one vector per layer that points toward "harmful" and away from
  "harmless" - Lesson 06 proves it actually means that, before Module 03 does anything to it.

### 06 - Sanity-checking a direction
- **Concept**: projection as a detector - project a held-out prompt's residual onto the
  direction and it should score higher for harmful prompts than harmless ones, at the layers
  where the concept is actually represented (usually middle layers, not the first or last
  few); why you always validate on held-out data, the same reason it matters in any ML
  workflow.
- **Example**: plot (as printed numbers, not a chart) the projection scores for the 4 + 4
  held-out prompts across a few candidate layers.
- **Practice**: implement `projection_scores`; the checker asserts a real separation margin
  between held-out harmful and harmless scores on the model's strongest layer.
- **Summary**: you've now found a direction and proven it means what you think it means -
  Module 03 is entirely about what to do with it.

## Dependencies

04 -> 05 -> 06, strictly; all of Module 02 depends on `get_residuals_per_prompt` from
Lesson 02. The direction produced here is reused unchanged through Module 05.

## Misconceptions to hit head-on

- "Compute the direction from one really clear example instead of averaging many." (A single
  prompt's activation carries prompt-specific noise the mean cancels out - this is the same
  reason you don't train a classifier on one example.)
- "Every layer should show the same separation." (Early layers represent surface tokens, not
  yet concepts; late layers are close to output formatting. The direction is real but its
  *strength* varies by layer - Lesson 12 is built entirely around this fact.)
- "If it works on the training prompts, it works." (Lesson 06 exists specifically to catch a
  direction that memorized the training set instead of generalizing.)
