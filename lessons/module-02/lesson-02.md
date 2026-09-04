# 05 - Computing the direction

You have two mean vectors per layer now - `mean_harmful` and `mean_harmless`, each
`[num_layers + 1, hidden_size]`. Lesson 03's `mean_difference` already told you how to combine
two group means into one vector: subtract them. Today you do exactly that, at every layer of a
real model, and add one more step on top - normalization - that the rest of this course leans
on hard enough to earn its own lesson.

## The direction, in one line

    direction = normalize(mean_harmful - mean_harmless)   # per layer, independently

`mean_harmful - mean_harmless` is a vector that points from "where harmless prompts tend to
sit" toward "where harmful prompts tend to sit," at a given layer. That's already a usable
direction - projecting a new prompt's residual onto it (Lesson 06) would already show some
separation. So why not stop there?

## Order matters

`mean_a - mean_b`, not `mean_b - mean_a` - the subtraction order isn't arbitrary bookkeeping,
it fixes the sign of the resulting direction. `compute_direction(mean_a, mean_b)` is defined
generically (the function itself doesn't know which group is "harmful"), but every call site in
this course passes the harmful mean first: `compute_direction(mean_harmful, mean_harmless)`.
That convention is what makes the direction point *toward* harmful and *away from* harmless -
and it's exactly what Lesson 06 relies on when it expects harmful prompts to score *higher* on
`v @ direction`, not lower. Swap the argument order once, anywhere, and every downstream sign
convention flips with it: "high projection score" would suddenly mean harmless instead of
harmful, silently, with no error to catch it. Keep the convention consistent and you never have
to think about it again; get it backwards once and every later lesson's numbers look
inverted for a reason that has nothing to do with a bug in those lessons.

## Why normalize

Lesson 02 pointed out that the residual stream's norm grows across layers - later layers have
accumulated more additions, so their vectors tend to be longer. That growth has nothing to do
with how *strongly* a layer represents "refusal" - it's a property of the architecture, not of
the concept. If you left `mean_harmful - mean_harmless` un-normalized, a late layer's raw
difference vector might be numerically huge just because everything at that layer is huge, and
an early layer's might be numerically tiny for the same structural reason - even if the early
layer represents the refusal concept *more* cleanly, relatively speaking. Comparing raw
difference magnitudes across layers would be comparing apples measured in different units.

Normalizing each layer's direction to unit length - `direction[i].norm() == 1` for every
layer `i` - throws away that architecture-driven scale and keeps only *which way* each layer's
difference points. This buys you something concrete two modules from now: once directions are
unit vectors, "how strongly should ablation act at this layer" becomes a separate,
explicit number you choose (Lesson 12), instead of something silently baked into whichever
layer happened to produce a bigger raw difference. Separating "which way" from "how much" is
the entire reason this step exists - it is not a formality.

`torch.nn.functional.normalize(x, dim=-1)` does this per-vector along the last dimension. For
a `[num_layers + 1, hidden_size]` tensor, `dim=-1` means: normalize each layer's
`hidden_size`-long vector independently, so layer 3's direction has no influence on the length
of layer 20's direction. That independence matters - each layer gets its own unit vector,
computed only from that layer's own mean difference.

## Why layer 0 is a zero vector

Run the full computation across all layers, not just the middle one, and you'll find
something that looks like a bug and isn't: `direction[0]` - the direction at the raw
embedding layer, before any attention has run - comes out as the zero vector, not a unit
vector. `F.normalize` leaves an exact zero vector as zero (there's nothing to normalize), so
this shows up as `direction[0].norm() == 0.0` while every other layer is `1.0`.

The cause is `encode_chat`'s chat template, not a mistake in your code. Every prompt is
formatted with `add_generation_prompt=True`, which appends a fixed suffix telling the model
"now generate the assistant's reply" - and that suffix's last token is the **same token** for
every prompt, regardless of what the prompt asked. At layer 0, before any attention block has
run, the residual stream at that last position is *only* that one token's embedding - nothing
about the actual prompt has been mixed in yet. So `mean_harmful[0]` and `mean_harmless[0]` are
computed from an identical embedding every time, the two means are identical,
`mean_harmful[0] - mean_harmless[0]` is exactly zero, and there's no direction to normalize.

From layer 1 onward, at least one attention pass has happened, and attention is exactly the
mechanism that lets the last position's residual stream start incorporating information from
every earlier token - which is where the prompt's actual content, and therefore any
"harmful vs. harmless" signal, first shows up. This is a real, structural fact about
decoder-only chat models, not an edge case to code around: layer 0 of this course's directions
is expected to be zero, and Lesson 06 onward only ever looks at layer 1 and later anyway.

## Checking your own normalization

Lesson 03 already gave you the tool to sanity-check this without a model: cosine similarity.
A unit vector's cosine similarity with itself is always 1.0 (it's `v . v / (norm(v) *
norm(v))`, and for a unit vector `norm(v) == 1`, so this is just `v . v`, which for a unit
vector also equals 1.0). Its cosine similarity with an unrelated random vector should sit near
0.0, because in a high-dimensional space, two random vectors are very likely close to
perpendicular by chance - there's no reason for them to agree. Neither check tells you the
direction is *meaningful* (that's Lesson 06's job) - only that the normalization arithmetic
itself is correct.

## Do this

1. In a Python shell, compute the direction at one middle layer and run both sanity checks:

       from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN, num_layers
       from lib.direction import get_residuals_mean
       import torch
       import torch.nn.functional as F

       mean_harmful = get_residuals_mean(HARMFUL_TRAIN)
       mean_harmless = get_residuals_mean(HARMLESS_TRAIN)

       layer = num_layers() // 2
       diff = mean_harmful[layer] - mean_harmless[layer]
       direction = F.normalize(diff, dim=0)

       print(direction.norm().item())                                  # 1.0
       cos_self = (direction @ direction).item()
       print(cos_self)                                                 # 1.0

       random_vec = torch.randn_like(direction)
       cos_random = (direction @ random_vec / (direction.norm() * random_vec.norm())).item()
       print(cos_random)                                                # close to 0.0

2. Open `lab/exercises/lesson_05.py` and implement `compute_direction(mean_a, mean_b)`:
   subtract the two mean tensors and normalize the result along the last dimension, so every
   layer gets its own unit-length direction. Both inputs and the output are
   `[num_layers + 1, hidden_size]`.

3. Grade it:

       bash lab/lab.sh check 05

   The checker compares your output to a reference implementation on real means from
   `HARMFUL_TRAIN` / `HARMLESS_TRAIN`, and separately checks that every layer from 1 onward has
   unit norm - not just the one layer you happened to test by hand - and that layer 0 is the
   zero vector explained above.

## Hints

- `mean_a - mean_b`, then `F.normalize(diff, dim=-1)` - two lines, no loop over layers needed.
  `dim=-1` is what makes normalization apply per-layer instead of across the whole flattened
  tensor.
- Don't normalize `mean_a` and `mean_b` separately before subtracting - that changes the
  result and is not what `compute_direction` does. Subtract first, normalize the difference.
- If `direction.norm(dim=-1)[1:]` isn't all-ones, check you passed `dim=-1` and not the
  default (`F.normalize`'s default is `dim=1`, which is wrong here unless your tensor happens
  to be 2-D with layers as the first axis - which it is, so this specific mistake is easy to
  make silently. Be explicit.). Index `0` is supposed to be zero - see above - don't "fix" it.

## Solution

    import torch
    import torch.nn.functional as F


    def compute_direction(mean_a: torch.Tensor, mean_b: torch.Tensor) -> torch.Tensor:
        diff = mean_a - mean_b
        return F.normalize(diff, dim=-1)

## Summary

You now have one unit vector per layer, computed from real activations, that points from
"harmless" toward "harmful." Nothing so far has proven that this vector actually means what
its construction suggests - it's entirely possible it captured something specific to the 20
training prompts of each class instead of the general concept. Lesson 06 is the check: project
prompts this direction has *never seen* onto it, and see whether the harmful ones score higher
than the harmless ones.
