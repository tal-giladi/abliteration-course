# 03 - Behavior lives in a direction, not a neuron

Here's a claim this whole course rests on: when a model has learned a concept - "this is a
request I was trained to refuse" - that concept is represented, approximately, as a single
**direction** in the residual stream. Not one neuron lighting up. Not a special flag bit.
A direction: a vector that a huge number of "this is a refusal-worthy prompt" activations all
point roughly the same way along, and that "this is a fine prompt" activations don't.

This is called the **linear representation hypothesis**, and you don't need to take it on
faith - Module 02 proves it, on a real model, with your own code. Today's lesson is just
making sure the vector math under that claim is second nature before a real 1024-dimensional
version of it shows up.

## Why "direction" and not "neuron"

If concepts were stored one-per-neuron, you could find "the refusal neuron" by checking which
single dimension of the residual stream is high for refused prompts and low for others. In
real trained networks, this essentially never works cleanly - concepts are represented as
**combinations** of many dimensions at once, i.e. as directions that aren't aligned with any
single axis. The tool for asking "how much does this vector point along that direction" isn't
"read one coordinate" - it's a **dot product**.

## The dot product as a detector

For two vectors `a` and `b`, the dot product `a . b = sum(a_i * b_i)` is large and positive
when they point the same way, near zero when they're perpendicular, and large and negative
when they point opposite ways. If `d` is a *unit* vector (length 1) representing "the refusal
direction," then `v . d` tells you how much of `v` points along it - and that number is
exactly what Lesson 06 will use to detect refusal-worthy prompts.

## Cosine similarity vs. dot product - they are not the same thing

`a . b` depends on both direction *and* magnitude - a long vector pointing slightly off from
`d` can produce a bigger dot product than a short vector pointing exactly along `d`. **Cosine
similarity** removes magnitude from the picture:

    cosine_similarity(a, b) = (a . b) / (norm(a) * norm(b))

This is always between -1 (opposite) and 1 (identical direction), regardless of how long `a`
and `b` are. You'll use dot products directly once directions are normalized to unit length
(Lesson 05 does this on purpose, specifically so raw dot products become meaningful without
a separate normalization step every time) - but until then, comparing two directions honestly
means cosine similarity, not a raw dot product.

## Do this

No model, no download - this lesson is pure vector arithmetic on small synthetic examples,
deliberately, so you can check every step by hand before Module 02 does the same math against
1024 real dimensions.

1. In a Python shell:

       import torch
       a = torch.tensor([3.0, 4.0])
       b = torch.tensor([6.0, 8.0])   # same direction as a, different magnitude
       c = torch.tensor([-4.0, 3.0])  # perpendicular to a

       print((a @ b).item())                                   # dot product: not 1, even though same direction
       print((a @ b / (a.norm() * b.norm())).item())            # cosine similarity: 1.0
       print((a @ c / (a.norm() * c.norm())).item())            # cosine similarity: ~0.0

2. Open `lab/exercises/lesson_03.py` and implement two functions:
   - `cosine_similarity(a, b) -> float`
   - `mean_difference(group_a, group_b) -> Tensor` - given two batches of vectors (shape
     `[n, dim]` each), return the difference of their means, `mean(group_a) - mean(group_b)`
     (this is the exact operation Lesson 05 applies to real activations - practice it here
     first, on numbers you can verify by hand).

3. Grade it:

       bash lab/lab.sh check 03

## Hints

- `torch.linalg.norm(v)` or `v.norm()` - either works for a vector's length.
- `mean_difference` operates on *batches* - `group_a.mean(dim=0)` collapses the batch
  dimension, leaving one vector.
- Don't normalize inside `mean_difference` - that's a separate, deliberate step Lesson 05
  adds on top, not something this function should do silently.

## Solution

    import torch

    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a @ b / (a.norm() * b.norm())).item()

    def mean_difference(group_a: torch.Tensor, group_b: torch.Tensor) -> torch.Tensor:
        return group_a.mean(dim=0) - group_b.mean(dim=0)

## Summary

A direction is just a vector that means something because of what it's aligned with, and a
dot product is how you measure alignment. Module 02 runs exactly this math - a mean
difference, then a dot product - against real activations from a real model, and proves the
result actually separates harmful prompts from harmless ones.
