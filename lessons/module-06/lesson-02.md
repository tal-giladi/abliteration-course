# 17 - Reading `model.py::abliterate()` for real

This is deliberately the hardest lesson in the course. Not because the ideas are new - every
piece of `heretic`'s real `Model.abliterate()` is something you've already built, across
Modules 03 and 04 - but because production code buries those ideas under everything a toy
script doesn't need: dtype juggling, quantization, an actual `peft` library instead of two
tensors you hand-rolled, and a strength curve with four parameters instead of a flat number.
The skill this lesson teaches isn't new math. It's reading dense code by first finding your own
code inside it.

Everything below is read directly from `heretic`'s real `src/heretic/model.py` - not
paraphrased from documentation, not guessed at. Open your own `lab/lib/ablate.py` next to it
if you have a local checkout; the rest of this lesson assumes you're looking at both.

## The one-line summary, and why it's still true

`Model.abliterate()` does exactly what `lib/ablate.py`'s `merge_into_model()` does: for a
chosen direction, compute a rank-1 correction and write it into the attention output
projection and MLP down projection of every layer, weighted by how strongly that layer should
be ablated. Every difference below is *how carefully* it does that, not *what* it does.

## `AbliterationParameters` is Lesson 12's strength curve, written down

Your toy `merge_into_model(model, directions, weights)` takes `weights` as a flat
`list[float]` - one number per layer, and Lesson 12 has you compute that list however you like.
The real version doesn't take a list. It takes an `AbliterationParameters` dataclass per
component:

    @dataclass
    class AbliterationParameters:
        max_weight: float
        max_weight_position: float
        min_weight: float
        min_weight_distance: float

and turns it into a weight *per layer* inside `abliterate()` itself:

    distance = abs(layer_index - params.max_weight_position)
    if distance > params.min_weight_distance:
        continue
    weight = params.max_weight + (distance / params.min_weight_distance) * (
        params.min_weight - params.max_weight
    )

Read that as a shape, not an equation: a triangular kernel, peaking at `max_weight` at layer
`max_weight_position`, falling off linearly to `min_weight` over `min_weight_distance` layers
in each direction, and *skipping the layer entirely* (`continue`) once you're further than
`min_weight_distance` away - which is why heretic's ablation is usually concentrated around the
middle-to-late layers instead of touching every layer equally. This is the exact question
Lesson 12 asks you to answer by hand ("why not every layer gets the same weight") - the real
answer is: because a four-parameter kernel searches that shape automatically instead of you
picking numbers.

## `get_abliterable_components()` is your hardcoded `o_proj`/`down_proj`, generalized

Your `merge_into_model` reaches directly into `layer.self_attn.o_proj.weight` and
`layer.mlp.down_proj.weight`, because `Qwen3-0.6B`'s architecture is fixed and known. The real
`get_layer_modules()` tries roughly a dozen different attribute paths per layer, each wrapped
in `with suppress(Exception): try_add(...)` - `self_attn.o_proj`, `linear_attn.out_proj` (for
hybrid models like Qwen3.5's linear-attention layers), `mlp.down_proj`, MoE expert lists
(`mlp.experts[i].down_proj`), `block_sparse_moe.experts[i].w2`, and several more. Every
`try_add` that doesn't match that architecture just silently does nothing. `get_abliterable_components()`
then unions whatever component names were actually found across every layer of the loaded
model. This is what "heretic supports most dense models, several MoE architectures, and some
hybrid models" (from its own README) costs in code: not smarter math, just a longer list of
"where might the residual stream get written to" for architectures you'll never personally
touch in this course.

## `peft.LoraConfig` is your `lora_rank1_factors()`, installed as a real adapter

`lib/ablate.py`'s `lora_rank1_factors(W, direction)` returns two plain tensors - `A` and `B` -
that satisfy `W + B @ A == ablate_matrix_output(W, direction)`. The real version does the same
factorization, but instead of handing you tensors to keep track of yourself, it installs them
as an actual `peft` LoRA adapter on the model:

    self.peft_config = LoraConfig(
        r=lora_rank,               # 1, unless row_normalization == "full"
        target_modules=target_modules,
        lora_alpha=lora_rank,      # apply the adapter at full strength
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    self.model = get_peft_model(self.model, self.peft_config)

and then, inside `abliterate()`, for every module it found:

    lora_A = (v @ W).view(1, -1)          # Lesson 09's A = d @ W
    lora_B = (-weight * v).view(-1, 1)    # Lesson 09's B = -d, scaled by this layer's weight
    ...
    weight_A.data = lora_A.to(weight_A.dtype)
    weight_B.data = lora_B.to(weight_B.dtype)

Line up `lora_rank1_factors`'s `A = (d @ W).unsqueeze(0)` and `B = (-d).unsqueeze(1)` against
those two lines. It's the identical formula. The only real addition is `weight` - your toy
version always ablates at full strength (`weight=1.0` baked into `ablate_matrix_output`'s
default); the real version multiplies `B` by whatever this layer's strength-curve weight came
out to, which is how a `max_weight` of, say, `0.6` produces a partial ablation instead of an
all-or-nothing one. The reason `heretic` uses a real `peft` adapter instead of raw tensors like
your course does: a `PeftModel` can be reset to identity in one call (`torch.nn.init.zeros_(lora_B.weight)`)
between search trials, which is exactly how `reset_model()`'s fast path works - your course
never needed that because Lesson 14's search loop just re-derives new tensors each trial
instead of reusing a live model object.

## The dtype upcast Module 03 never needed

Right before computing `lora_A`/`lora_B`, the real code does this:

    if quant_state is None:
        W = base_weight.to(torch.float32)
    else:
        W = bnb.functional.dequantize_4bit(base_weight.data, quant_state).to(torch.float32)

Your course's `lib/common.py` sets `DTYPE = torch.float32` once, globally, and every tensor in
this course has been float32 from the start - there was never a upcast to write, because there
was never a downcast to undo. A real model loaded in `bfloat16` (the common case - see Lesson
16's `dtypes` list) has roughly 7-8 bits of mantissa precision. Computing a dot product and a
rank-1 outer product in that precision, then writing the result back into a `bfloat16` weight,
compounds rounding error in a way that's invisible on any single number but measurably degrades
the model. Upcasting to float32 for the arithmetic and downcasting only the final result
(`.to(weight_A.dtype)`) is the standard fix - and if the base weight is 4-bit quantized (`bnb`),
it has to be *dequantized* to a real float representation before any of this math can happen at
all, which `bnb.functional.dequantize_4bit` does using the `quant_state` bitsandbytes stored at
load time.

## `residual_directions` is your `compute_direction()`, computed once per run

In `main.py`, before the search loop starts:

    residual_directions = F.normalize(bad_means - good_means, p=2, dim=1)

That's `direction.compute_direction(bad_means, good_means)`, called once, up front - not once
per trial. Every trial in the search loop reuses these same per-layer directions; what changes
between trials is only the `AbliterationParameters` (the strength curve) and, optionally, which
single direction gets used for every layer instead of each layer using its own (`direction_index`,
a float that can *interpolate* between two adjacent layers' directions via `math.modf` - a
continuous generalization of "pick one direction and use it everywhere" that Lesson 06's
sanity check treats as fixed per layer).

## `Model.get_residuals_mean` - same name, same job

Not every real name needs translating. `heretic`'s `Model.get_residuals_mean(prompts)` and your
`direction.get_residuals_mean(prompts)` share not just a job but a literal name - both average
`get_residuals_per_prompt`-style output over a prompt set to produce one `[num_layers + 1,
hidden_size]` tensor. The real version streams the sum in float64 on CPU batch-by-batch instead
of materializing every residual tensor at once (`lib/direction.py`'s version calls
`get_residuals_per_prompt(...).mean(dim=0)`, which keeps every prompt's full residual tensor in
memory simultaneously) - a memory optimization for datasets with hundreds of prompts, not a
different computation.

## Do this

1. Open `src/heretic/model.py` in your local `heretic-llm` install (find it with
   `lab/.venv/*/site-packages/heretic/model.py`, or read it on GitHub at
   `p-e-w/heretic`) side by side with `lab/lib/ablate.py` and `lab/lib/direction.py`.
2. Open `lab/exercises/lesson_17.py` and fill in `MAPPING` - a dict of 8 real
   `model.py`/`main.py` names to a one- or two-sentence explanation of which toy-course
   function or concept does the same job, and how they differ. The 8 real names to map are
   listed in the exercise stub's docstring.
3. Grade it:

       bash lab/lab.sh check 17

   The checker doesn't grade your prose for style - it checks that you named the real toy-course
   equivalent for each one, the same way a reviewer would skim your answer for "did they find
   the right piece of code," not "is this beautifully written."

## Hints

- You don't need `heretic` installed to do this exercise - it's reading and writing text, not
  running any code. Lesson 16's install is enough to have the real source available locally if
  you want it, but the source excerpts above are complete for every mapping you need.
- For the `AbliterationParameters` mapping, name the specific toy parameter it replaces (the
  `weights` argument to `merge_into_model`), not just "Lesson 12."
- For `get_abliterable_components`, either `o_proj` or `down_proj` is a fine anchor - the point
  is recognizing that the real version *discovers* what your toy version *assumes*.
- Two of the 8 real names both map to the same toy function (`lora_rank1_factors`) - that's not
  a mistake in the exercise, it's because `LoraConfig` sets up the *container* the real factors
  get installed into, while the `lora_A`/`lora_B` assignment is where the *math itself* happens.

## Solution

    MAPPING = {
        "Model.abliterate": (
            "merge_into_model() in lib/ablate.py — same job (write a rank-1 correction into "
            "o_proj/down_proj for every layer), generalized over dtype, quantization, and "
            "discovered components instead of two hardcoded ones."
        ),
        "AbliterationParameters": (
            "The weights argument to merge_into_model() — a flat per-layer list in the toy "
            "course, replaced by a 4-parameter triangular kernel (max_weight, "
            "max_weight_position, min_weight, min_weight_distance) that generates that list."
        ),
        "get_abliterable_components": (
            "The hardcoded self_attn.o_proj / mlp.down_proj lookups inside merge_into_model — "
            "the real version discovers component names per architecture instead of assuming "
            "them, so it works on models this course's fixed Qwen3-0.6B target never has to "
            "handle."
        ),
        "LoraConfig": (
            "lora_rank1_factors() in lib/ablate.py — same rank-1 factorization, but installed "
            "as a real peft adapter on the model instead of returned as plain A/B tensors, so "
            "it can be reset to identity between search trials."
        ),
        "lora_A / lora_B assignment in Model.abliterate": (
            "lora_rank1_factors()'s A = (d @ W).unsqueeze(0) and B = (-d).unsqueeze(1) — "
            "identical formula, with B additionally scaled by this layer's strength-curve "
            "weight instead of always being applied at full strength."
        ),
        "residual_directions in main.py::run": (
            "direction.compute_direction(mean_a, mean_b) — same normalized difference of "
            "means, computed once before the search loop instead of once per trial."
        ),
        "Model.get_residuals_mean": (
            "direction.get_residuals_mean() — same name, same job (average last-token "
            "residuals over a prompt set); the real version streams the sum in float64 to "
            "avoid holding every prompt's full residual tensor in memory at once."
        ),
        "W = W.to(torch.float32) in Model.abliterate": (
            "No toy equivalent needed — lib/common.py hardcodes DTYPE = torch.float32 for "
            "every tensor in this course, so there's never a lower-precision weight to "
            "upcast before doing the projection math, or downcast afterward."
        ),
    }

## Summary

Every piece of `abliterate()` you just read is something you already built - a strength curve,
a component lookup, a rank-1 factorization, a mean-difference direction - wrapped in exactly
the kind of production concerns (precision, architecture coverage, resettable state) that a
14-lesson toy course correctly skips until now. That recognition is the actual skill this
lesson was for: the next time you open unfamiliar transformer-tooling code, "wrapped in
production concerns" is what you should expect to find, not a wall of unrelated math. Lesson 18
uses that same reading skill to make one small, real, verified change to this codebase.
