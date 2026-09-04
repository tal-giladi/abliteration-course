# 09 - Why a rank-1 LoRA instead of a weight edit

`ablate_matrix_output` and `ablate_matrix_input` both overwrite a weight matrix in place -
that's what `merge_into_model` in `lab/lib/ablate.py` does, with `W.copy_(...)`. It works, but
it's destructive: once `o_proj.weight` has been overwritten, the only way back to the original
model is reloading it from disk. Module 05 is going to try hundreds of different `(layer,
weight)` combinations to find the ones that actually work - and reloading a several-hundred-
megabyte model from disk between every single trial is not a workflow anyone would tolerate.
This lesson gets you the same edit without that cost.

## Delta instead of edit

Instead of overwriting `W`, keep `W` untouched and store the *change* separately:

    ΔW = W' - W = -d (d^T W)

If you can compute `ΔW` cheaply and attach it at the moment of use - `y = Wx + ΔW x` - then
nothing about the base model's weights ever changes on disk or in memory. Attach `ΔW` to try
an ablation; detach it to get the exact original model back; swap in a different direction or
weight by attaching a different `ΔW`. This attach/detach pattern is a **LoRA adapter** (Low-
Rank Adaptation) - a technique originally built for fine-tuning, where you don't want to
maintain a separate full copy of the model's weights for every variant you've trained.

## Why "rank-1" isn't a compromise here

For fine-tuning, a LoRA's rank `r` is a hyperparameter you choose as a size/quality trade-off:
the "ideal" weight delta a full fine-tune would produce could need any rank to represent
exactly, so a small `r` is an *approximation* - you're deliberately giving up some expressive
power for a adapter that's cheap to store and swap.

That is not the situation here, and it's worth being precise about why. `ΔW = -d (d^T W)` is
an **outer product** of two vectors: `d` (shape `[out_features]`) and `d^T W` (shape
`[in_features]`). An outer product `a ⊗ b` produces a matrix whose every column is a scalar
multiple of `a` - by construction, its columns all live on a single line through the origin,
which means the matrix has **rank exactly 1**, always, no matter what `a` and `b` are. Lesson
08's edit was never going to need more than rank 1 to represent exactly, because the edit
*is* an outer product before you ever go looking for a low-rank version of it. There's nothing
to approximate.

## The factorization

`B @ A`, for `B` a column `[out_features, 1]` and `A` a row `[1, in_features]`, produces a
matrix whose `(i, j)` entry is `B_i * A_j` - exactly the shape of an outer product. Matching
that against `ΔW`'s `(i, j)` entry, `-d_i * (d^T W)_j`, gives the factorization directly:

    B = -d              (as a column, [out_features, 1])
    A = d^T W           (as a row,    [1, in_features])

    B @ A  =  (-d) ⊗ (d^T W)  =  -d (d^T W)  =  ΔW

No fitting, no optimization loop, no approximation error beyond ordinary floating point - `B`
and `A` are read off `ΔW`'s own two factors directly, because `ΔW` already had exactly that
shape. `W + B @ A` and `ablate_matrix_output(W, direction)` are the same matrix, not two
matrices that happen to be close.

## Why this is worth the trouble

Module 05's search loop needs to try a strength for one layer, measure the result, and move on
- over and over, hundreds of times. With `merge_into_model`, that means: reload the model,
edit the weights, run the model, decide the result was mediocre, reload the model again for
the next trial. With a rank-1 LoRA: compute `(A, B)` once per candidate (a handful of small
matrix operations, not a model reload), attach it, run the model, detach it, try the next
candidate against the same still-untouched base weights. The base model is loaded exactly
once, for the entire search. This is also, concretely, what `heretic` itself does - it's not
a simplification this course introduces for teaching purposes and drops later; Lesson 17 shows
you the real `peft`-based adapter code this factorization plugs into.

## Seeing it work

    import torch
    from lib.ablate import ablate_matrix_output, lora_rank1_factors

    torch.manual_seed(0)
    W = torch.randn(4, 6)
    d = torch.tensor([1.0, 0.0, 0.0, 0.0])

    A, B = lora_rank1_factors(W, d)
    print(A.shape, B.shape)                        # torch.Size([1, 6]) torch.Size([4, 1])

    delta_via_lora  = B @ A
    delta_direct    = ablate_matrix_output(W, d) - W
    print(torch.allclose(delta_via_lora, delta_direct, atol=1e-6))   # True

    print(torch.allclose(W + B @ A, ablate_matrix_output(W, d), atol=1e-6))   # True

The last line is the actual claim of this lesson made concrete: reconstructing `W` from its
LoRA-adapted form gives you back precisely the matrix Lesson 08 would have produced by editing
`W` directly - not an approximation of it.

## Do this

1. Run the snippet above.

2. Open `lab/exercises/lesson_09.py` and implement `lora_rank1_factors(W, direction) ->
   tuple[Tensor, Tensor]`, returning `(A, B)` with `A: [1, in_features]` and
   `B: [out_features, 1]`.

3. Grade it:

       bash lab/lab.sh check 09

   The checker compares your `(A, B)` against `lib.ablate.lora_rank1_factors`, then does the
   actual proof independently of that comparison: it reconstructs `W + B @ A` using **your**
   `A` and `B` and checks it equals `ablate_matrix_output(W, direction)` computed straight from
   `W` - confirming the factorization is exact, not just that you matched a reference
   function's output. It repeats both checks against the real `o_proj.weight` shape.

## Hints

- `direction` needs the same internal normalization as Lessons 07 and 08 - `d^T W` (and
  therefore `A`) is wrong if `d` isn't unit length first.
- `A = (d @ W).unsqueeze(0)` - `d @ W` alone gives a 1-D tensor of shape `[in_features]`;
  `unsqueeze(0)` turns it into the required `[1, in_features]` row.
- `B = (-d).unsqueeze(1)` - same idea, turning `[out_features]` into `[out_features, 1]`.
- Matrix multiplication order matters: `B @ A` (column times row) gives the full
  `[out_features, in_features]` outer product you want. `A @ B` would instead contract the
  matching `1` dimensions and give you a `[1, 1]` scalar - if your shapes don't match `W`,
  check this first.

## Solution

    import torch
    import torch.nn.functional as F


    def lora_rank1_factors(W: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = F.normalize(direction, dim=-1)
        A = (d @ W).unsqueeze(0)
        B = (-d).unsqueeze(1)
        return A, B

## Summary

You now have the exact operation Heretic hangs off a real `peft` LoRA adapter during its
search, in three pieces: Lesson 07's projection, Lesson 08's application of it to the two
matrices per layer that actually write to the residual stream, and this lesson's proof that
the resulting edit is already, exactly, a rank-1 delta you can attach and detach at will.
Module 04 puts all three to work against the real model for the first time - starting with the
simplest of the three, a live forward hook that ablates every generation without touching a
single weight, before Lesson 11 makes an edit permanent with the exact math from Lesson 08,
and Lesson 12 varies how strongly each layer gets ablated instead of applying one flat weight
everywhere.
