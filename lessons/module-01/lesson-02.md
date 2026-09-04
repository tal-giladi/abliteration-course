# 02 - Reading hidden states across layers

Last lesson you pulled `out.hidden_states` once and looked at the last entry. This lesson is
about the other `num_hidden_layers` entries, and about building the one function almost
every later lesson in this course calls.

## The object every lesson shares

`output_hidden_states=True` gives you a tuple of `num_hidden_layers + 1` tensors, each shaped
`[batch, seq_len, hidden_size]`. Think of it as a photograph of the residual stream taken at
every checkpoint - before layer 0 (the raw embedding), after layer 0, after layer 1, ... ,
after the last layer. `torch.stack` turns that tuple into one tensor,
`[batch, layers + 1, seq_len, hidden_size]`, and from here on this course calls that shape a
**residual tensor**. Every direction, every projection, every ablation in this course
operates on residual tensors with this exact shape convention - layer index first, always
`num_layers + 1` of them.

## Why the last token, specifically

A decoder-only model predicts the *next* token from the residual stream at the *current
last* position - everything the model knows about "what should come next" is concentrated
there. Earlier positions' final-layer streams matter for how attention built up the last
position's stream, but they are not what the model reads to decide what to say. So when this
course asks "what does the model's internal state look like for this prompt," it means,
specifically: the residual stream at the last token, at whichever layer you're asking about.

This is also why Lesson 01 had you left-pad. With left-padding, the last column
(`[:, :, -1, :]`) is *always* the last real token for every sequence in a batch, no matter how
short one prompt is next to another. Right-padding would put pad tokens at the end of the
shorter sequences, and `[:, :, -1, :]` would silently grab garbage for them.

## Norm growth is a real, visible pattern

If you ran a few different prompts through the shell from Lesson 01, you may have noticed the
norm at layer 0 is small and the norm at the last layer is much larger. This is a well-known,
consistent property of trained transformers - the stream's magnitude tends to grow as more
layers add to it, because addition accumulates. It's not a bug, and it's not something you
need to correct for; you'll see it again in Lesson 05 as the reason directions get
**normalized** before they're used for anything.

## Do this

1. In a Python shell, capture and compare two very different prompts:

       from lib.common import encode_chat, load_model
       import torch

       model = load_model()
       for prompt in ["What is the capital of France?", "asdkfj qwoeiru"]:
           inputs = encode_chat([prompt])
           out = model(**inputs, output_hidden_states=True)
           norms = [h[:, -1, :].norm().item() for h in out.hidden_states]
           print(prompt, "->", [round(n, 1) for n in norms])

   Look at how the norm profile differs - a coherent prompt and gibberish leave different
   traces across the layers, even though you haven't looked for anything specific yet.

2. Now build the reusable version. Open `lab/exercises/lesson_02.py` and implement
   `get_residuals_per_prompt(prompts, batch_size=8)`, matching the shape and batching
   described above: given a list of prompts, return a tensor
   `[len(prompts), num_layers + 1, hidden_size]` - the last-token residual stream, at every
   layer, for every prompt, batched so you're not forwarding one prompt at a time.

3. Grade it:

       bash lab/lab.sh check 02

   The checker runs your function against the course's real model on a handful of prompts and
   compares the result to a reference implementation - not a fixed expected file, since the
   "right answer" is whatever the real model actually produces.

## Hints

- Loop over `prompts` in chunks of `batch_size`, call `encode_chat` on each chunk, and
  concatenate the per-chunk results at the end with `torch.cat(..., dim=0)`.
- `torch.stack(out.hidden_states, dim=1)` turns the tuple into one
  `[batch, layers+1, seq, hidden]` tensor - stacking along `dim=1`, not `dim=0`, is what keeps
  `batch` as the first axis.
- Slice the last token with `[:, :, -1, :]` on that stacked tensor.
- Decorate your function with `@torch.no_grad()` (or wrap the body in `with torch.no_grad():`)
  - you're not training anything, and skipping gradient tracking is both faster and avoids
  memory you don't need.

## Solution

    import torch
    from lib.common import encode_chat, load_model

    @torch.no_grad()
    def get_residuals_per_prompt(prompts, batch_size=8):
        model = load_model()
        rows = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            inputs = encode_chat(batch)
            out = model(**inputs, output_hidden_states=True)
            stacked = torch.stack(out.hidden_states, dim=1)
            rows.append(stacked[:, :, -1, :])
        return torch.cat(rows, dim=0)

## Summary

Every lesson from here forward operates on this one object - a per-layer, per-prompt snapshot
of the residual stream, always shaped `[n_prompts, num_layers + 1, hidden_size]`. Module 02
starts running this function on prompt sets specifically designed to disagree with each other.
