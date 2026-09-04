# Module 04 quiz - Building the pipeline

Seven questions. Answers and explanations at the bottom - try all seven first.

---

**1.** A forward hook registered on `model.model.layers[i]` can change what the model
generates without editing a single file on disk. In one or two sentences, explain how that's
possible - what is the hook actually intercepting, and when does its effect end?

**2.** `ablation_forward_hook` is registered on `model.model.layers[i]` using `directions[i +
1]`, not `directions[i]`. Explain the off-by-one - what does `directions[i]` actually
correspond to instead?

**3.** `live_ablation`'s hook-removal has to happen inside a `finally` block (or a context
manager's `__exit__`), not just "at the end of the function." What could go wrong if it were
only at the end?

**4.** A colleague says: "a live hook and `merge_into_model` are two different abliteration
techniques - one edits the model, one doesn't." Correct them - what's actually the same
between the two, and what's actually different?

**5.** Lesson 11's checker doesn't just compare in-memory tensors after `merge_into_model`
runs - it also saves the model to disk with `save_pretrained`, reloads it fresh with
`from_pretrained`, and generates from that. What would the tensor-only comparison fail to
prove that the save/reload step does prove?

**6.** A colleague sets `max_weight=1.0` and a very large `min_weight_distance` (say, `10.0`)
in `strength_curve`, reasoning "stronger ablation on every layer can only help." What actually
happens to the returned curve when `min_weight_distance` is that large, and why is "ablate
everything at full strength" not obviously the right call even before you consider that?

**7.** In `merge_into_model`, a checkpoint with `weight == 0` is skipped with `continue`
rather than still calling `ablate_matrix_output(W, d, weight=0.0)`. Both produce the same
resulting tensor - so why does it matter which one the code actually does?

---

## Answers

**1.** A forward hook runs after the layer's own `forward()` produces its output, and if the
hook returns a value, that value replaces the layer's output for everything downstream - so
the hook is intercepting the residual-stream contribution that specific layer just computed,
before the next layer (or the final logits) ever sees it. Its effect lasts exactly as long as
the hook stays registered on that module - remove the handle (or exit the context manager that
holds it) and the very next forward pass runs with nothing intercepted, because nothing about
the model's own code or weights ever changed.

**2.** A `directions` tensor is indexed the same way `output_hidden_states` is: index 0 is the
embedding output, before any decoder layer has run; index `i` is the residual stream *after*
decoder layer `i - 1` has added its contribution. A hook on `model.model.layers[i]`
intercepts that layer's output - which is checkpoint `i + 1` in that indexing, not checkpoint
`i`. Using `directions[i]` would apply the direction computed for the stream *before* layer
`i` ran to the stream *after* it ran - the wrong slice, one layer early.

**3.** If the code inside the `with` block raises - a bad prompt, an unexpected exception, a
typo in the caller - and removal only happens as the last line of the function, that line
never runs. The hooks stay registered on the model for the rest of the process, quietly
changing every subsequent forward pass through it, with no error and no obvious sign anything
is still attached. Putting removal in a `finally` (or `__exit__`) guarantees it runs whether
the block finished cleanly or not.

**4.** What's the same: the actual math. Both are `v' = v - (v . d) d` (scaled to a matrix,
where needed) applied to exactly the same places - the outputs that write to the residual
stream. What's different is only *when* the cost of computing that is paid and *where* the
result lives. A hook recomputes and reapplies the projection on every forward pass, on
whatever the layer happens to output that call, and it's gone the instant the handle is
removed. `merge_into_model` applies the identical projection once, directly to the weight
tensors that would have produced that output in the first place, so every future forward pass
already reflects it with no runtime hook required - at the cost of no longer being trivially
reversible.

**5.** The in-memory tensors after `merge_into_model` only prove the function correctly edited
the specific Python objects it was handed in that process. It says nothing about whether a
saved copy of that model - a `.safetensors` file some other process loads fresh, with no
memory of your hook-free Python session - actually has the edit. `save_pretrained` +
`from_pretrained` + generate is the only way to prove the ablation is a property of the file on
disk, not an artifact of the objects still sitting in your interpreter's memory.

**6.** With `max_weight_position` at some fixed spot and `min_weight_distance = 10.0`, every
checkpoint's normalized distance `t = dist / 10.0` stays far below `1.0` for any `dist` that's
realistically at most `1.0` (the whole stack is only 1.0 units wide) - so `t` never clamps to
the floor anywhere, and the curve stays close to `max_weight` across the *entire* stack instead
of tapering off. That's effectively uniform full-strength ablation again, which is exactly the
thing Lesson 12 exists to move away from: Lesson 06 already showed some layers separate
harmful from harmless prompts far more cleanly than others, so ablating every layer at full
strength removes more than the target concept from the layers where the direction wasn't
cleanly "about" refusal in the first place - trading refusal reduction for coherence you didn't
mean to spend.

**7.** `ablate_matrix_output(W, d, weight=0.0)` does produce the same numerical result as
leaving `W` untouched, but it still does the work to get there - normalizing `d`, computing
`d @ W`, building the outer product, subtracting a zero-scaled version of it, and then
`copy_`-ing the (unchanged) result back into the parameter. `continue` skips all of that for a
checkpoint the caller has explicitly said should get zero ablation, which is both the cheaper
path and the one that says what's actually happening more directly - "this checkpoint is
untouched" is clearer as "skip it" than as "apply a projection scaled by zero."
