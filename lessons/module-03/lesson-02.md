# 08 - From vectors to weight matrices

Lesson 07 zeroed a direction out of one vector, and proved it. But a model doesn't hand you
"one vector" to fix - it produces a different residual-stream activation for every possible
input, and there's no way to enumerate them all and ablate each one by hand. This lesson
applies the same subtraction to something that reaches every one of them at once: a weight
matrix.

## Which matrices actually write to the stream

Every `nn.Linear` layer in the model computes `y = W @ x`: read a vector `x`, produce a vector
`y`. `W` has shape `[out_features, in_features]`. That's true of every linear projection in a
decoder layer - but only *some* of those `y`s end up added into the residual stream. Recall
Lesson 01: a layer's contribution to the stream comes from a residual connection,
`stream = stream + f(stream)` - and `f` here is a whole sub-block of computation, most of
which never touches the stream directly.

Inside the attention block, `q_proj`, `k_proj`, and `v_proj` each read the current stream and
produce queries, keys, and values - intermediate values, consumed by the attention computation
itself, never added back anywhere. Only after attention has mixed information across token
positions does `o_proj` take *that* result and produce the vector that actually gets added to
the stream. Inside the MLP block, `gate_proj` and `up_proj` similarly read the stream and
produce intermediate values that get combined and gated; only `down_proj`'s output is what the
residual connection adds back. Two matrices per decoder layer write to the stream -
`self_attn.o_proj` and `mlp.down_proj` - and the rest only ever read from it.

This matters concretely: if you ablated `q_proj` instead of `o_proj`, you'd be projecting the
direction out of a vector that gets fed straight into an attention score computation and never
reaches the residual stream in that form at all. The edit would do nothing measurable, for a
very unglamorous reason - you'd be editing a matrix whose output isn't the thing you're trying
to change.

## The matrix form of Lesson 07's formula

You want: for **every** possible input `x`, the output `y = Wx` should satisfy `y . d = 0`.
Not "for the `x` we happened to test" - for all of them, structurally, because you can't
enumerate every `x` a real model will ever see. Start from Lesson 07's formula and apply it to
`y` symbolically:

    y' = y - (y . d) d
       = Wx - (d . Wx) d
       = Wx - (d^T W x) d          [d . Wx is the same number as d^T W x]
       = Wx - d (d^T W x)
       = Wx - d (d^T W) x          [x doesn't depend on d^T W, so it factors out]
       = (W - d (d^T W)) x

`x` was arbitrary, and it factored cleanly out of every term - so the matrix in front of it,
`W - d (d^T W)`, is a new weight matrix that produces an ablated output for *any* `x` you feed
it, without needing to know `x` in advance:

    W' = W - d (d^T W)

`d^T W` is a `[in_features]` row vector - "how much does each *output* direction of `W`
already point along `d`, summed across everything `W` reads." `d (d^T W)` is an outer product,
`[out_features, in_features]`, same shape as `W` - the matrix that, subtracted off, cancels
exactly the part of every output that pointed along `d`. In code this is
`torch.outer(d, d @ W)`.

## The embedding matrix is the same idea from the other side

`embed_tokens.weight` is shaped `[vocab_size, hidden_size]`, and it isn't computing `y = Wx`
for some input at all - it's a lookup table. Row `i` **is** the initial residual stream for
token id `i` - literally `out.hidden_states[0]` from Lesson 01, before any decoder layer has
run. There's no `y = Wx` to project through here; each row already *is* a residual-stream
vector, directly, so ablating this matrix means applying Lesson 07's formula to every row at
once, not to every possible output of a linear map:

    E' = E - (E d) d^T

`E @ d` is a `[vocab_size]` column - "how much does each row already point along `d`" - and
`(E @ d) d^T`... well, `torch.outer(E @ d, d)` gives the `[vocab_size, hidden_size]` matrix
whose row `i` is `(row_i . d) d`, exactly the per-row component Lesson 07 subtracts. Same
formula, same proof, just applied down a matrix instead of across one - `ablate_matrix_output`
projects what a matrix can *produce*; `ablate_matrix_input` projects what a matrix's rows
*already are*.

## `weight`: how much to apply

Both functions take a `weight` argument, `1.0` by default: `weight = 1.0` removes the
direction completely (`W' = W - d(d^T W)`); `weight = 0.0` changes nothing; anything in
between is a partial ablation, `W - weight * d(d^T W)`. You won't need this for a while - every
example in this lesson uses the default - but Lesson 12 tunes it per layer instead of applying
a flat `1.0` everywhere, so it's part of the signature from the start.

## Seeing it work

    import torch
    from lib.ablate import ablate_matrix_output, ablate_matrix_input

    torch.manual_seed(0)
    W = torch.randn(4, 6)              # a synthetic "hidden_size=4, in_features=6" matrix
    d = torch.tensor([1.0, 0.0, 0.0, 0.0])

    W_prime = ablate_matrix_output(W, d)
    for _ in range(3):
        x = torch.randn(6)             # a different random input each time
        y = W_prime @ x
        print((y @ d).item())          # ~0.0 every time - for ANY x, not just one

    E = torch.randn(8, 4)              # synthetic "vocab_size=8, hidden_size=4" table
    E_prime = ablate_matrix_input(E, d)
    print(E_prime @ d)                 # length-8 vector, every entry ~0 - every row lost it

The first loop is the point: three completely different random inputs, and every one produces
an output with zero component along `d`. That's what "ablated the matrix" buys you over
ablating individual vectors - the property holds for inputs you'll never even see.

## Do this

1. Run the snippet above.

2. Open `lab/exercises/lesson_08.py` and implement both `ablate_matrix_output(W, direction,
   weight=1.0)` and `ablate_matrix_input(E, direction, weight=1.0)`.

3. Grade it:

       bash lab/lab.sh check 08

   The checker compares both functions to `lib.ablate`'s reference on small synthetic
   matrices, then loads the real course model (cached from `lab/lab.sh up`, no download) and
   runs both functions once against `model.model.layers[0].self_attn.o_proj.weight` and
   `model.model.embed_tokens.weight` - the actual shapes Module 04 will edit for real - with a
   synthetic direction sized to the model's real `hidden_size`. No forward pass happens, so
   this stays fast even with the model loaded.

## Hints

- `torch.outer(a, b)` builds the `[len(a), len(b)]` matrix of every pairwise product - exactly
  what both formulas need, just with the arguments in a different order for each function.
- For `ablate_matrix_output`: `d` lives in `W`'s *output* space (`hidden_size`), so it's
  `torch.outer(d, d @ W)` - `d` first.
- For `ablate_matrix_input`: `d` lives in `E`'s *row* space, and `E @ d` is "how much each row
  already points along `d`" - `torch.outer(E @ d, d)` - `d` second this time. If your shapes
  come out transposed from `W`/`E`, this is almost always the argument order to check.
- Don't forget `weight` - both formulas multiply the whole correction term by it, not just
  part of it: `W - weight * torch.outer(...)`.
- Normalize `direction` first, same reasoning as Lesson 07 - the derivation above only holds
  when `d . d = 1`.

## Solution

    import torch
    import torch.nn.functional as F


    def ablate_matrix_output(W: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        d = F.normalize(direction, dim=-1)
        return W - weight * torch.outer(d, d @ W)


    def ablate_matrix_input(E: torch.Tensor, direction: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        d = F.normalize(direction, dim=-1)
        return E - weight * torch.outer(E @ d, d)

## Summary

Ablating a model was never going to mean touching every one of its parameters - it means
finding the handful of matrices whose outputs (or, for the embedding table, whose rows) are
the residual stream itself, and applying Lesson 07's projection there. For Qwen3-0.6B that's
exactly three matrices per relevant layer worth of code: `o_proj`, `down_proj`, and
`embed_tokens`. What you have now is destructive, though - it overwrites `W` in place, which
means trying a different direction or a different `weight` means starting from an unedited
copy again. Next lesson expresses this exact same edit as something you can attach and detach
instead.
