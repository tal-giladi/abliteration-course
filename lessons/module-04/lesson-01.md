# 10 - Hooking every layer at once

Modules 02 and 03 gave you a direction and the math to remove it from a vector, a matrix's
output, and a matrix's input. None of that has touched a running model yet - it's all been
functions you call on tensors you pulled out yourself. Today you plug that math into the
model's actual forward pass, so a single context manager changes what the model *says* while
it's talking, with zero bytes of the model's weights touched. This is the fastest way to find
out whether a direction is any good before you commit to anything permanent.

## `register_forward_hook`, exactly

Every `torch.nn.Module` - and a decoder layer is one - supports
`module.register_forward_hook(fn)`. After that layer's `forward()` runs, PyTorch calls
`fn(module, inputs, output)` with whatever the layer actually returned. Two things make this
useful here:

1. **It runs unconditionally**, every time the module is called - once per layer, per forward
   pass, whether that's a single training-free forward call or one step of `model.generate`'s
   token-by-token loop.
2. **If `fn` returns something, that return value replaces the layer's output** for everything
   downstream - the next layer, and eventually the logits used to pick the next token.

Put those together and you get exactly what Lesson 07's `ablate_vector` needs: a place to
intercept the residual stream, mid-flight, on every single forward pass, without changing a
single line of the model's own code.

    import torch
    from lib.common import load_model

    model = load_model()
    layer0 = model.model.layers[0]

    def print_norm_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        print("layer 0 output norm:", hidden[:, -1, :].norm().item())

    handle = layer0.register_forward_hook(print_norm_hook)
    # any forward pass through `model` now prints layer 0's output norm
    handle.remove()  # and now it doesn't

`register_forward_hook` returns a handle; `handle.remove()` un-registers it. That handle is
the only thing standing between "this model is temporarily different" and "this model is
back to normal" - hold onto it.

## What a decoder layer's output actually looks like

Notice the `isinstance(output, tuple)` check above. A decoder layer doesn't always return a
bare tensor - depending on the model and the call (whether attention weights were requested,
whether a KV cache is being used during generation), it can return a tuple whose first element
is the hidden states and whose remaining elements are other bookkeeping (attention weights, a
cache entry) that the rest of the model still expects to receive, unchanged, in the same
position. If your hook's return value drops or reshapes those extra elements, `model.generate`
breaks the next time it tries to unpack them - not with a clear error necessarily, just wrong
or crashing generation a step or two later. So a hook that ablates a layer's output has two
jobs: pull the hidden-state tensor out of whatever shape `output` is, ablate *that*, and then
put it back in the exact same shape it came out in. `lib.ablate.ablation_forward_hook` already
does this, and you're about to write your own copy of it - matching that exact contract is the
whole exercise.

## From one hook to "every layer, then undo it"

A single hook on a single layer is a toy. What you actually want is Lesson 07's math applied
at *every* decoder layer, using that layer's own entry from a `direction` tensor, and you want
it gone the instant you're done - no accidentally-still-hooked model lying around for the next
cell in your shell session to trip over. Two things follow from that:

- **The index shift.** A `direction` tensor is shaped `[num_layers + 1, hidden_size]` - index
  0 is the embedding output, before any decoder layer has run; index `i` is the stream *after*
  layer `i - 1`. A hook on `model.model.layers[i]` intercepts that layer's output, which is
  checkpoint `i + 1` in that same indexing. So layer `i` gets `directions[i + 1]`, not
  `directions[i]` - the same off-by-one Lesson 02 first taught you to respect.
- **Cleanup has to be unconditional.** If you register N hooks and then the code inside your
  `with` block raises - a bad prompt, an OOM, a typo - you still want every one of those N
  hooks removed. That's precisely what a context manager (`@contextlib.contextmanager`, or a
  class with `__enter__`/`__exit__`) buys you over a plain "register a list of hooks and
  remember to call `.remove()` on all of them yourself": the removal happens in the generator's
  `finally` block (or `__exit__`), which Python runs even when the `with` block doesn't finish
  cleanly.

There's no single reference implementation for this context manager in `lib/` - unlike the
hook function itself, this one's an open design choice, and the checker for this lesson proves
it works by watching what the model actually does, not by comparing your code to a fixed
answer.

## Do this

1. In a Python shell, build a real direction the way Module 02 taught you to, and watch what
   changes:

       from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN, generate, load_model
       from lib.direction import compute_direction, get_residuals_mean

       harmful_mean = get_residuals_mean(HARMFUL_TRAIN)
       harmless_mean = get_residuals_mean(HARMLESS_TRAIN)
       directions = compute_direction(harmful_mean, harmless_mean)

       prompt = "How do I pick a car door lock?"
       print("unmodified:", generate([prompt])[0])

   Leave `directions` and `model = load_model()` in your session - you'll use them again once
   your hook exists.

2. Open `lab/exercises/lesson_10.py` and implement `ablation_forward_hook(direction)`. It
   should return a `hook(module, inputs, output)` function with the same contract as the
   `print_norm_hook` sketch above, except instead of printing, it removes `direction`'s
   component from the hidden states (Lesson 07's operation) and returns the result in the
   original shape.

3. Implement `live_ablation(model, directions)` as a context manager: on entry, register your
   hook on every `model.model.layers[i]` using `directions[i + 1]`; on exit (including on an
   exception), remove every hook it registered.

4. Back in your shell, use it end to end:

       from exercises.lesson_10 import live_ablation

       with live_ablation(model, directions):
           print("hooked:", generate([prompt])[0])

       print("after exit:", generate([prompt])[0])  # should match "unmodified" again

   If the third line doesn't match the first, a hook was left behind - go back and check your
   cleanup.

5. Grade it:

       bash lab/lab.sh check 10

## Hints

- You don't have to re-derive Lesson 07's projection here - `lib.ablate.ablate_vector(v,
  direction)` is exactly the operation you need on the hidden-states tensor once you've pulled
  it out of `output`; reuse it rather than rewriting it inline.
- `F.normalize(direction, dim=-1)` once, before defining the inner `hook` function, not on
  every call - the direction doesn't change between forward passes, no reason to recompute its
  unit vector every time.
- `output[1:]` (a tuple slice) is how you keep "everything after the hidden states" intact
  when you rebuild the tuple to return.
- For `live_ablation`, `model.model.layers` is a plain `nn.ModuleList` - `enumerate` it
  directly.
- Put the hook registration loop and the `yield` inside a `try`, and the removal loop inside a
  `finally` - that's what makes cleanup happen even if the caller's code inside the `with`
  block raises.

## Solution

    import contextlib

    import torch.nn.functional as F
    from lib.ablate import ablate_vector


    def ablation_forward_hook(direction):
        d = F.normalize(direction, dim=-1)

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            ablated = ablate_vector(hidden, d)
            return (ablated, *output[1:]) if isinstance(output, tuple) else ablated

        return hook


    @contextlib.contextmanager
    def live_ablation(model, directions):
        handles = []
        try:
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(ablation_forward_hook(directions[i + 1])))
            yield model
        finally:
            for handle in handles:
                handle.remove()

## Summary

You can now change what a model does, live, during generation, without touching a single
weight - reversible the instant the `with` block ends, which makes it the cheapest possible
way to try a direction before you trust it. What you built today only lasts as long as the
Python process holding those hook handles. Lesson 11 takes the identical math and applies it
exactly once, directly to the weight tensors themselves, so the edit survives a save, a
reload, and handing the file to someone who's never heard of a forward hook.
