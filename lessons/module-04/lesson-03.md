# 12 - Choosing which layers and how strongly

`merge_into_model` has taken a `weights` argument since the moment you wrote it - Lesson 11
just always passed the default, one uniform `1.0` for every checkpoint. That default was the
right call for proving the mechanism works. It is the wrong call for actually shipping an
ablated model, and Lesson 06 already told you why, even if it didn't look like it at the time:
projecting held-out prompts onto a direction and checking separation, layer by layer, showed
that some layers separate harmful from harmless prompts cleanly and others barely separate them
at all. A direction that means "refusal" strongly at layer 14 might mean almost nothing in
particular at layer 2 - early layers are still doing a lot of work that has nothing to do with
the refusal concept yet (surface-level token and syntax processing, mostly), so subtracting a
full-strength "refusal direction" from them removes real signal the model needs for other
things, not just the behavior you're trying to erase. Ablate every layer at strength 1.0 and
you're not just erasing refusal - you're erasing it plus whatever else that direction happened
to correlate with at the layers where it wasn't really "about" refusal in the first place. That
shows up as coherence loss: a model that answers everything and answers it worse.

The fix isn't "pick one better uniform number." It's "stop using a uniform number."

## Four knobs, one shape

Heretic - the real tool this course is building a small version of - doesn't expose "the
ablation strength." It exposes four numbers that together describe a *curve* across the layer
stack:

- **`max_weight`** - the strongest ablation applied anywhere, at whichever layer needs it most.
- **`max_weight_position`** - *where* along the stack that peak sits, expressed as a fraction
  from 0 (the embedding output) to 1 (the last layer) - not a layer index, because that lets
  the same setting make sense across models with different layer counts.
- **`min_weight`** - the floor: however far a checkpoint is from the peak, ablation strength
  never drops below this.
- **`min_weight_distance`** - how quickly strength falls off as you move away from the peak,
  in the same 0-to-1 units as `max_weight_position`. Small distance = a narrow spike that hits
  the floor fast on either side; large distance = a broad, gentle curve that stays near
  `max_weight` almost everywhere.

Put together, these four numbers describe a shape that's high in the middle of the stack and
low near the edges by default - not because "the middle is always right" (Module 05 is what
actually finds good values instead of assuming), but because it's a reasonable place to start
looking, and because it turns "ablation strength" from a single on/off switch into something
with enough range to describe both "barely touch anything" and "hit every layer hard," and
everything between.

## A simple, exact falloff

There's more than one reasonable curve shape you could draw through those four numbers. This
course picks the simplest one that's still exactly checkable: **linear, clamped at the floor.**
For checkpoint `i` (out of `num_layers + 1` total, indices `0..num_layers`), its position along
the stack is `pos = i / num_layers`. Its distance from the peak is `dist = abs(pos -
max_weight_position)`. Normalize that distance against `min_weight_distance` and clamp it to
`1.0`:

    t = min(dist / min_weight_distance, 1.0)

and interpolate linearly from `max_weight` down to `min_weight` using `t`:

    weight = max_weight - (max_weight - min_weight) * t

At the peak (`dist = 0`), `t = 0` and you get `max_weight` exactly. Once you're
`min_weight_distance` away or farther, `t` clamps to `1.0` and you get `min_weight` exactly, no
matter how much farther you go. In between, it's a straight line. That's a triangular spike,
not a smooth bell curve - and that's the point: it's simple enough that you can compute exact
expected outputs by hand for a handful of `(num_layers, max_weight, max_weight_position,
min_weight, min_weight_distance)` combinations, which is exactly what today's checker does.
(Heretic's actual default curve shape is worth comparing against once you've built this one -
Module 06 opens the real source and shows you.)

## Do this

1. In a shell, sketch a curve for a small `num_layers` and eyeball the shape before you write
   any code against a real model:

       # by hand, for num_layers=4, peak in the middle, floor at the edges:
       # positions:      0.00  0.25  0.50  0.75  1.00
       # dist from 0.5:  0.50  0.25  0.00  0.25  0.50
       # if min_weight_distance = 0.5, t = dist / 0.5:
       #                 1.00  0.50  0.00  0.50  1.00
       # weight = 1.0 - (1.0 - 0.0) * t:
       #                 0.00  0.50  1.00  0.50  0.00

   Work through what happens if you move `max_weight_position` to `0.0` (peak at the very
   start) or make `min_weight_distance` tiny (a narrow spike that hits the floor almost
   immediately) before opening the exercise file - you'll recognize the shape in the checker's
   test cases.

2. Open `lab/exercises/lesson_12.py` and implement `strength_curve(num_layers, max_weight,
   max_weight_position, min_weight, min_weight_distance) -> list[float]`, returning exactly
   `num_layers + 1` values using the formula above.

3. Once it passes, try feeding its output straight into last lesson's function on a real
   model - this is what the four knobs are actually for:

       from lib.common import HARMFUL_TRAIN, generate, load_model
       from lib.direction import compute_direction, get_residuals_mean
       from lib.common import HARMLESS_TRAIN
       from exercises.lesson_11 import merge_into_model
       from exercises.lesson_12 import strength_curve
       import copy

       directions = compute_direction(get_residuals_mean(HARMFUL_TRAIN), get_residuals_mean(HARMLESS_TRAIN))
       n = load_model().config.num_hidden_layers
       curve = strength_curve(n, max_weight=1.0, max_weight_position=0.5, min_weight=0.1, min_weight_distance=0.3)

       curved_model = copy.deepcopy(load_model())
       merge_into_model(curved_model, directions, weights=curve)
       # generate from curved_model the same way Lesson 11 did, and compare against a
       # uniform-1.0 merge - Module 05 is where "compare" becomes a real measurement.

4. Grade it:

       bash lab/lab.sh check 12

## Hints

- Guard `min_weight_distance <= 0` before you divide by it - treat a checkpoint exactly at the
  peak (`dist == 0`) as `t = 0` regardless, and anything off-peak as `t = 1` (fully floored),
  so a degenerate `min_weight_distance` can't crash the function or silently blow up `t`.
- `pos = i / num_layers` needs float division - in Python 3, `/` already gives you that (`//`
  would not).
- Build the list with a simple loop or list comprehension over `range(num_layers + 1)` - there's
  no tensor involved here at all, this function doesn't touch the model or `torch`.
- Double-check the off-by-one: the function returns `num_layers + 1` values for an
  `num_layers`-layer model, matching `merge_into_model`'s `weights` argument exactly - the same
  shape every direction tensor in this course has used since Lesson 05.

## Solution

    def strength_curve(num_layers, max_weight, max_weight_position, min_weight, min_weight_distance):
        curve = []
        for i in range(num_layers + 1):
            pos = i / num_layers
            dist = abs(pos - max_weight_position)
            if min_weight_distance <= 0:
                t = 0.0 if dist == 0 else 1.0
            else:
                t = min(dist / min_weight_distance, 1.0)
            curve.append(max_weight - (max_weight - min_weight) * t)
        return curve

## Summary

Ablation strength is a per-layer dial now, not a single switch - four numbers describe a whole
curve across the stack instead of one flat multiplier applied everywhere. What you don't have
yet is a principled way to *pick* those four numbers, or to know whether a given curve actually
did better than uniform strength beyond eyeballing one generated response. That's Module 05:
scoring refusal and coherence together as real numbers, and searching over exactly the four
knobs you just built the shape for, instead of guessing them by hand.
