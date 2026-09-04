# 11 - Making the edit permanent

A hook is great for deciding *whether* a direction works. It's a bad way to ship one. Every
process that wants the ablated behavior would need to know to register your hook, in the right
place, with the right direction, before doing anything else - and if that process is a serving
framework you don't control, or a teammate who just wants to `from_pretrained` a model and
generate, that's not a reasonable thing to ask. This lesson turns the exact same math into
something that needs none of that: a model whose *weights* already encode the ablation, so
loading it plainly and generating from it is enough.

## The same math, applied once, into the weights that write it

Lesson 08 established which two matrices, inside a decoder layer, actually write to the
residual stream: `self_attn.o_proj.weight` (attention's contribution) and `mlp.down_proj.weight`
(the MLP's). Everything a layer adds to the stream comes from one of those two places - which
means "remove `direction` from everything this layer could ever write" is exactly
`ablate_matrix_output` applied to both of them, once, and `ablate_matrix_input` applied once to
`embed_tokens.weight` for the very first checkpoint. That's the entire content of
`merge_into_model`: loop over every layer, edit those two matrices in place with
`Tensor.copy_()` inside a `torch.no_grad()` block, and you're done - no return value, because
there's nothing left to return. The model you passed in *is* the ablated model now.

This is worth sitting with, because it's the direct answer to a misconception that's easy to
carry out of Lesson 10: a live hook and a permanent merge are not two different techniques that
happen to produce similar results. They are the *same* projection - `v' = v - (v . d) d`,
scaled up to a matrix - applied at two different times. The hook applies it fresh, on every
forward pass, to whatever the layer happens to output that call. The merge applies it once, to
the weight matrices themselves, so every future forward pass produces an output that's already
missing that component - because the matrix that would have written it no longer can. Nothing
about *what* gets removed changes. Only *when* and *where* you pay the cost of removing it
does.

## What actually changes operationally

With a hook, generation with the edit active needs: the original model, your hook function, a
direction, and code that registers the hook before every call that should be affected. Forget
any one of those and you're back to the unmodified model, silently.

With a merge, generation needs: the saved model. That's the whole list. `save_pretrained`
writes the (now-ablated) weight tensors to a `.safetensors` file the normal way; anyone who
later calls `AutoModelForCausalLM.from_pretrained` on that directory gets a model that refuses
less, with no idea that abliteration was ever involved - because from its perspective, nothing
was. This is also the trade you're accepting: a hook is trivially reversible (remove the
handle); a merge is not, unless you kept an unedited copy of the weights around separately. You
are not un-ablating a saved, merged model - you're loading a different, unedited copy of it.

## Proving it survived, not just checking it happened

It would be easy to write `merge_into_model`, run it, inspect the in-memory tensors, see they
changed, and call it done. That only proves the function edited the tensors it was handed - it
says nothing about whether what you'd actually *ship* (a directory on disk) behaves correctly
once it's loaded back the way a real consumer would load it: fresh process, fresh
`from_pretrained` call, no hooks, no leftover Python objects from your session. So this
lesson's checker does both: a fast weight-tensor comparison against the reference
implementation, and a slower, real round trip - save to disk, reload, generate, check the
response isn't a refusal. The second one is doing real I/O and a real model load on top of the
first one's math check; it's slower for exactly that reason, and that's expected on this
course's small model.

## Do this

1. In a shell, build the same direction you used in Lesson 10 (or reuse it if your session is
   still open), and look at one weight tensor before touching anything:

       import copy
       from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN, load_model
       from lib.direction import compute_direction, get_residuals_mean

       harmful_mean = get_residuals_mean(HARMFUL_TRAIN)
       harmless_mean = get_residuals_mean(HARMLESS_TRAIN)
       directions = compute_direction(harmful_mean, harmless_mean)

       model = load_model()
       before = model.model.layers[0].self_attn.o_proj.weight.clone()

   Note the `copy.deepcopy` you'll need in a moment: `load_model()` is `@lru_cache`d, so
   calling it a second time hands you back the *same* object, not a second independent model.
   `copy.deepcopy(load_model())` is how you get a second, independent instance cheaply, without
   downloading or re-initializing anything - the model is small enough (~0.6B parameters) that
   deepcopy on CPU is fast.

2. Open `lab/exercises/lesson_11.py` and implement `merge_into_model(model, directions,
   weights=None)`: mutate `model.model.embed_tokens.weight` in place with
   `ablate_matrix_input`, then for every decoder layer, mutate `self_attn.o_proj.weight` and
   `mlp.down_proj.weight` in place with `ablate_matrix_output`, all inside `torch.no_grad()`,
   using `Tensor.copy_()` so the edit lands in the existing tensor rather than replacing it
   with a new one the rest of the model doesn't see.

3. Verify the merge actually changed something, and that it's the *right* something:

       your_model = copy.deepcopy(model)
       merge_into_model(your_model, directions)
       print(torch.allclose(before, your_model.model.layers[0].self_attn.o_proj.weight))  # False
       print(torch.allclose(model.model.layers[0].self_attn.o_proj.weight, before))        # True - `model` untouched

4. Grade it:

       bash lab/lab.sh check 11

   This check is noticeably slower than the ones before it - it does a real `save_pretrained` /
   `from_pretrained` round trip through a temp directory (`lab/.cache/tmp-lesson-11`, which
   `lab/.cache/` already keeps out of git) on top of the weight comparison. That's expected.

## Hints

- Don't re-derive the projection math here - `ablate_matrix_output` and `ablate_matrix_input`
  are exactly Lesson 08's functions, already correct in `lib/ablate.py`; import and reuse them.
  This lesson is about the *loop and the in-place mutation*, not the projection itself.
- `weights = weights or [1.0] * (len(layers) + 1)` gives you the "ablate everything at full
  strength" default this lesson uses - Lesson 12 is what makes that list interesting.
- A `weight` of exactly `0` should skip that checkpoint entirely (`continue`) rather than call
  `ablate_matrix_output` with `weight=0.0` - both give the same tensor back, but skipping is
  the cheaper and more obviously-correct choice, and it's what the reference does.
- `tensor.copy_(new_value)` edits `tensor` in place; `tensor = new_value` would just rebind the
  local variable and leave the model's actual parameter untouched - a classic silent bug here.

## Solution

    import torch
    from lib.ablate import ablate_matrix_input, ablate_matrix_output


    def merge_into_model(model, directions, weights=None):
        layers = model.model.layers
        weights = weights or [1.0] * (len(layers) + 1)
        with torch.no_grad():
            emb = model.model.embed_tokens.weight
            if weights[0] != 0:
                emb.copy_(ablate_matrix_input(emb, directions[0], weights[0]))
            for i, layer in enumerate(layers):
                w = weights[i + 1]
                if w == 0:
                    continue
                d = directions[i + 1]
                o_proj = layer.self_attn.o_proj.weight
                o_proj.copy_(ablate_matrix_output(o_proj, d, w))
                down_proj = layer.mlp.down_proj.weight
                down_proj.copy_(ablate_matrix_output(down_proj, d, w))

## Summary

"Ablated" now means something you can hand someone as a file: a `.safetensors` checkpoint that
behaves differently, forever, with no runtime scaffolding, no hook, no Python session required
to keep it that way. Both this lesson and Lesson 10 used `weights`/`weight` at a flat 1.0 on
every layer - full strength, everywhere, because that was the simplest thing to prove the
mechanism worked. Lesson 12 is where you stop doing that: not every layer should lose the same
amount of the direction, and you're about to build the function that decides how much each one
does.
