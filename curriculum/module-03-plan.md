# Module 03 plan - The ablation trick

**Lessons 07-09. Prerequisite: Module 02. Produces: `lab/lib/ablate.py`'s vector- and
matrix-level projection functions, and the rank-1 factorization behind Heretic's LoRA trick.**

## Objectives

1. Derive and implement orthogonal projection: removing a direction from a vector.
2. Identify which weight matrices in a real transformer write to the residual stream, and
   apply the same projection to them.
3. Explain - and prove numerically - why expressing that edit as a rank-1 LoRA delta is
   equivalent to editing the weight directly, and why Heretic does the former.

## Lessons

### 07 - Orthogonal projection
- **Concept**: `v' = v - (v . d) d` for a unit vector `d` - the component of `v` along `d`,
  subtracted off; why the result is guaranteed orthogonal to `d` (a short, honest derivation,
  not just an assertion); this single line of algebra *is* abliteration - everything else in
  the course is about where and how carefully to apply it.
- **Example**: apply it to a synthetic vector with a known component along a known direction
  and watch that component disappear.
- **Practice**: implement `ablate_vector` in `lab/lib/ablate.py`; the checker verifies the
  output's dot product with the direction is ~0 to floating-point tolerance, for both a
  single vector and a batch.
- **Summary**: "remove a behavior" reduces, mathematically, to "delete one direction's worth
  of a vector" - deceptively small for what it does to a model.

### 08 - From vectors to weight matrices
- **Concept**: `nn.Linear`'s `y = W @ x` and which matrices' *outputs* land back in the
  residual stream - `self_attn.o_proj` (attention's contribution) and `mlp.down_proj` (the
  MLP's) - versus matrices that only read from it; the same projection applied to every
  possible output of `W` at once: `W' = W - d (d^T W)`; the embedding matrix `embed_tokens`
  as the same idea from the other side (its *rows*, not its outputs, live in the stream).
- **Example**: apply the matrix form to a small synthetic weight matrix and confirm every
  output it can produce is now orthogonal to the direction, for several random inputs.
- **Practice**: implement `ablate_matrix_output` and `ablate_matrix_input`; checked against
  the real shapes of `o_proj.weight`, `down_proj.weight`, and `embed_tokens.weight` on
  `Qwen/Qwen3-0.6B`.
- **Summary**: ablating a model doesn't mean editing every parameter - it means finding the
  handful of matrices that actually write to the stream, and projecting there.

### 09 - Why a rank-1 LoRA instead of a weight edit
- **Concept**: editing `W` in place is destructive and slow to iterate on (reload the model
  to try a different direction or strength); a LoRA adapter expresses a weight *delta*,
  `B @ A`, that can be attached, detached, or merged without touching the base weights; the
  factorization that makes this exact rather than approximate for a rank-1 edit:
  `-d(d^T W) = B @ A` where `B = -d`, `A = d^T W` - rank 1 is not a compromise here, it's
  the exact form the edit already had.
- **Example**: compute `B @ A` for a small synthetic `W` and direction, and show it matches
  `ablate_matrix_output(W, d) - W` element-for-element.
- **Practice**: implement `lora_rank1_factors`; the checker verifies `W + B @ A` equals
  `ablate_matrix_output(W, direction)` within tolerance, on the real `o_proj` weight shape.
- **Summary**: you now have the exact operation Heretic's `model.py` applies via a real
  `peft` LoRA adapter - Module 04 wires it into an actual forward pass and Module 06 shows
  you the production version.

## Dependencies

07 -> 08 -> 09, strictly; each lesson's function is used unmodified by the next. The
direction from Module 02 is the input to every exercise here.

## Misconceptions to hit head-on

- "Ablation zeroes out a neuron." (It removes one *direction's* worth of every output a
  matrix can produce - a structural change to what the matrix can express, not a value pinned
  to zero.)
- "You need calculus to derive this." (Orthogonal projection is a dot product and a
  subtraction - Lesson 07's entire derivation fits in four lines and uses nothing past a
  high-school vector course.)
- "LoRA is an approximation here, like it is for fine-tuning." (For a *rank-1* edit
  specifically, it's exact - Lesson 09's checker proves equality, not closeness, within
  floating-point tolerance.)
