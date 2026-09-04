# 07 - Orthogonal projection

Module 02 left you holding something real: a direction `d`, extracted from a real model's
activations, that separates prompts the model refuses from prompts it answers (Lesson 06
proved this by projecting held-out prompts onto it and watching them land on two different
sides). This lesson answers the question the entire course exists to answer: given a vector
that has some of `d` mixed into it, how do you remove exactly that, and nothing else?

## The projection you already know

Lesson 03 introduced the dot product as a detector: for a unit vector `d`, `v . d` tells you
how much of `v` points along `d`. That number isn't just a similarity score - it's a length.
Specifically, it's the length of the "shadow" `v` would cast onto the line through `d`. Scale
`d` by that length, `(v . d) d`, and you get an actual vector: the *component of `v` along
`d`* - a copy of `d`, stretched or shrunk to match how much of `v` was pointing that way.

That component is the part of `v` this lesson wants gone. Subtract it, and you're left with
whatever of `v` had nothing to do with `d`:

    v' = v - (v . d) d

This is **orthogonal projection**. It has a two-word name because it does two things at once:
it *projects* `v` onto the space perpendicular to `d` (throws away the `d`-flavored part), and
the result is guaranteed *orthogonal* to `d` (zero component left along it). "Guaranteed" is
doing real work in that sentence, and it's worth not taking on faith.

## The proof, in full

Assume `d` is a **unit vector**: `d . d = 1`. Take the dot product of `v'` with `d`:

    v' . d = (v - (v . d) d) . d
           = v . d  -  (v . d) (d . d)      [dot product distributes over subtraction]
           = v . d  -  (v . d) (1)          [because d . d = 1]
           = 0

That's the entire derivation. Four lines, nothing past distributing a dot product over
subtraction and substituting `d . d = 1`. No calculus, no gradients, nothing about neural
networks at all - this identity is true for *any* vector `v` and *any* unit vector `d`, in any
number of dimensions, whether `v` is a 2-element toy example or a 1024-dimensional residual
stream activation from a real transformer. That generality is exactly why this one line of
algebra is the whole trick behind abliteration: it doesn't need to know anything about
language models to work, which is what lets it get applied, unmodified, to a weight matrix
next lesson and to every layer of a real model in Module 04.

Notice where `d . d = 1` was load-bearing: if `d` weren't unit length, the middle term would
be `(v . d)(d . d)`, not `(v . d)`, and it wouldn't cancel the first term unless `d . d`
happened to equal 1. That's why every function in this course that takes a `direction`
argument normalizes it first, defensively, even if you're confident you already normalized it
upstream (Lesson 05 does normalize the direction it produces - but a function that only works
when its caller remembered to do that is a function waiting to fail silently). `ablate_vector`
below normalizes internally so it's correct no matter what the caller hands it.

## Seeing it work

    import torch
    from lib.ablate import ablate_vector

    d = torch.tensor([0.0, 5.0])   # length 5, not unit - ablate_vector normalizes it
    v = torch.tensor([7.0, 3.0])   # 3 units of v point straight along d's direction

    v_prime = ablate_vector(v, d)
    print(v_prime)                                       # tensor([7., 0.])
    print((v_prime @ torch.tensor([0.0, 1.0])).item())    # 0.0 - the d-component is gone

`v`'s first coordinate (perpendicular to `d`) survives untouched; its second coordinate (the
entire reason it had any overlap with `d`) is zeroed exactly. Nothing about `v`'s x-component
was touched, and nothing "close to zero because floating point" about the y-component either -
this is exact, up to the same floating-point tolerance as any other tensor arithmetic.

## The batch case

Real usage never ablates one vector at a time - Module 04 hooks every token position, every
layer, every forward pass. `ablate_vector` handles a batch `[n, hidden]` the same way it
handles a single vector `[hidden]`, because the formula doesn't change - only the bookkeeping
does. `v @ d` gives a single scalar for one vector, but a `[n]` vector of per-row dot products
for a batch (matrix-vector multiply, one dot product per row, for free). To subtract
`coeff * d` from every row, the coefficient needs an extra trailing dimension so it broadcasts
against `d`'s `[hidden]` shape: `coeff.unsqueeze(-1) * d` turns a `[n]` coefficient into
`[n, 1]`, which broadcasts against `[hidden]` to produce `[n, hidden]` - one scaled copy of `d`
per row, subtracted from that row.

## Do this

1. Run the shell snippet above and confirm it matches, then try a batch:

       import torch
       from lib.ablate import ablate_vector

       d = torch.tensor([0.0, 1.0, 0.0])
       v_batch = torch.tensor([[1.0, 2.0, 3.0], [5.0, -4.0, 1.0]])
       out = ablate_vector(v_batch, d)
       print(out)                       # middle column is 0 for every row
       print(out @ d)                   # tensor([0., 0.])

2. Open `lab/exercises/lesson_07.py` and implement `ablate_vector(v, direction)`: normalize
   `direction`, compute the coefficient, subtract - handling both the single-vector and
   batched shapes described above.

3. Grade it:

       bash lab/lab.sh check 07

   The checker compares your output to `lib.ablate.ablate_vector` on both a single vector and
   a batch, and - independently of that comparison - checks directly that your result's dot
   product with the normalized direction is zero to floating-point tolerance. Matching the
   reference isn't enough on its own; the orthogonality has to actually hold.

## Hints

- `torch.nn.functional.normalize(direction, dim=-1)` turns any nonzero vector into a unit
  vector in one call - this is what "normalize `direction` internally" means in code.
- `v @ d` is the right expression for *both* shapes - PyTorch's matmul already does the
  "batch of dot products" thing for you when `v` is 2-D and `d` is 1-D. You don't need a loop
  or `torch.sum(v * d, dim=-1)`, though that would also work.
- Use `v.dim()` to tell the two cases apart: `1` for a single vector, `2` for a batch. Only
  the batch case needs `coeff.unsqueeze(-1)`.
- If your orthogonality check fails by a suspiciously large amount (not just floating-point
  noise), the most common cause is forgetting to normalize `direction` before using it in the
  formula - re-read the "load-bearing" paragraph above.

## Solution

    import torch
    import torch.nn.functional as F


    def ablate_vector(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        d = F.normalize(direction, dim=-1)
        coeff = v @ d
        return v - coeff.unsqueeze(-1) * d if v.dim() > 1 else v - coeff * d

## Summary

"Remove a behavior" now has an exact, provable meaning: delete one direction's worth of a
vector, and the four-line proof above is the whole reason you can trust the result is really
gone, not just smaller. But a real model doesn't hand you one vector to fix - it can produce
an unbounded number of different activations, one for every possible input, and you can't
chase them down one at a time. Next lesson applies the exact same subtraction to something
with far more reach than a single vector: a weight matrix, so that *every* output it could
ever produce, for *any* input, loses the direction at once.
