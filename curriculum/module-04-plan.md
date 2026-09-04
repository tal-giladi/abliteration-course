# Module 04 plan - Building the pipeline

**Lessons 10-12. Prerequisite: Module 03. Produces: a real ablated model you can talk to,
first live and reversible, then permanently merged, then tunable per layer.**

## Objectives

1. Ablate a real model's generation live, using forward hooks, with zero weights changed.
2. Turn that live edit into a permanent one: merge into the weights, save, and reload.
3. Control ablation strength per layer instead of applying it uniformly everywhere.

## Lessons

### 10 - Hooking every layer at once
- **Concept**: `register_forward_hook` on a decoder layer intercepts its output before it's
  used by the next layer; registering the Lesson 07 projection as a hook on every layer
  ablates the *live* residual stream during generation without editing a single weight;
  why this is the fastest way to *try* a direction before committing to it.
- **Example**: generate a response to a harmful prompt from the unmodified model, then again
  with hooks registered, side by side.
- **Practice**: implement `ablation_forward_hook` in `lab/lib/ablate.py` and a context
  manager that registers/removes it across all layers; checked by generating a fixed prompt
  with and without hooks and asserting the outputs differ and the hooked one is not a refusal.
- **Summary**: you can now change a model's behavior *without changing the model* - reversible,
  cheap, and exactly what you want while you're still deciding on a direction.

### 11 - Making the edit permanent
- **Concept**: `merge_into_model` - the live hook's math, applied once to the actual weight
  tensors in place, so no hooks are needed afterward; `save_pretrained` /
  `from_pretrained` round-tripping a model whose weights now permanently encode the ablation.
- **Example**: merge, save to a temp directory, reload the saved model fresh (a new process
  state, no hooks), and generate the same prompt again.
- **Practice**: run `merge_into_model` on a copy of the model, save, reload, and the checker
  asserts the reloaded model's response to the fixed prompt is not a refusal - proving the
  edit survived a full save/reload cycle.
- **Summary**: "ablated" now means something concrete - a `.safetensors` file that behaves
  differently, forever, with no runtime scaffolding required.

### 12 - Choosing which layers and how strongly
- **Concept**: the `weight` parameter already threaded through `ablate_matrix_output` since
  Lesson 08, finally used for real; why full-strength ablation on every layer over-corrects
  (early layers carry more than just the target concept); a strength curve as a function of
  layer index - low near the edges, higher in the middle - mirroring the four knobs Heretic
  actually exposes (`max_weight`, `max_weight_position`, `min_weight`, `min_weight_distance`).
- **Example**: compare refusal rate on the held-out harmful set at uniform strength 1.0 vs. a
  simple bell-shaped strength curve.
- **Practice**: implement a `strength_curve(num_layers, max_weight, max_weight_position,
  min_weight, min_weight_distance) -> list[float]` function; checked against known
  input/output pairs and against the shape Heretic's `AbliterationParameters` expects.
- **Summary**: ablation isn't one on/off switch - it's a per-layer dial, and Module 05 is
  about finding good settings for that dial automatically instead of by hand.

## Dependencies

10 -> 11 -> 12. Lesson 11 reuses Lesson 09's `lora`-equivalent math conceptually but merges
directly (simpler, matches this module's "live -> permanent" arc); Lesson 12's strength
curve is the exact shape `merge_into_model`'s `weights` argument expects.

## Misconceptions to hit head-on

- "A forward hook and a weight edit are two different techniques." (Same math, two different
  places to apply it - live at inference time, or once into the weights. Lesson 11 exists to
  make that equivalence concrete.)
- "More ablation strength is always better." (Past a point it degrades coherence faster than
  it reduces refusals - Module 05 is what quantifies "past a point" instead of eyeballing it.)
- "Every layer contributes equally to refusal." (Lesson 06 already showed separation varies
  by layer; Lesson 12 is the direct consequence - strength should too.)
