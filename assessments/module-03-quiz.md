# Module 03 quiz - The ablation trick

Seven questions. Answers and explanations at the bottom - try all seven first.

---

**1.** A colleague says "ablation zeroes out the neuron that represents refusal." Correct them
in one or two sentences, using the word "direction."

**2.** Given a unit vector `d` (so `d . d = 1`) and any vector `v`, show in a few lines that
`v' = v - (v . d) d` satisfies `v' . d = 0`.

**3.** `ablate_vector`, `ablate_matrix_output`, and `ablate_matrix_input` all normalize
`direction` internally instead of trusting the caller to pass a unit vector already. Why does
that matter for the proof in question 2?

**4.** A Qwen3 decoder layer has five linear projections across its attention and MLP blocks:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. `merge_into_model`
only ever ablates two of these per layer. Which two, and why those specifically and not the
others?

**5.** `embed_tokens.weight` gets ablated with `ablate_matrix_input`, not
`ablate_matrix_output`, even though it's still just a big matrix inside the same model. What's
different about what its rows represent, compared to what `o_proj.weight`'s outputs represent?

**6.** "A rank-1 LoRA here is an approximation of the true weight edit, the same way a LoRA is
an approximation for fine-tuning." True or false - and be specific about what makes this
particular case different from a general fine-tuning LoRA.

**7.** You're about to try 200 different `(layer, weight)` combinations to find the best
ablation strength per layer (Module 05 automates exactly this). Why would you rather attach
and detach a rank-1 LoRA adapter for each trial than call `merge_into_model` and reload the
model fresh before every one?

---

## Answers

**1.** Ablation removes a *direction* - a specific combination of many dimensions of the
residual stream at once - not a value pinned to a single neuron. Real trained networks
essentially never store a concept in one coordinate; behavior is represented as directions
that don't line up with any single axis, which is exactly why the tool for finding and
removing one is a dot product and a subtraction, not "find the neuron and zero it."

**2.** `v' . d = (v - (v . d) d) . d = v . d - (v . d)(d . d) = v . d - (v . d)(1) = 0`, using
that the dot product distributes over subtraction and that `d . d = 1` because `d` is unit
length.

**3.** The middle step of the proof, `(v . d)(d . d) = (v . d)`, only holds because `d . d`
was assumed to equal exactly `1`. If `direction` weren't unit length, `d . d` would be some
other number, the two terms wouldn't cancel, and `v'` would be left with a nonzero,
magnitude-dependent component along `d` instead of being truly orthogonal to it. Normalizing
inside the function makes the guarantee hold regardless of what the caller passes in, instead
of quietly depending on the caller having done it correctly upstream.

**4.** `self_attn.o_proj` and `mlp.down_proj` - because those are the only two matrices per
layer whose *output* is what actually gets added into the residual stream via the residual
connection (attention's contribution and the MLP's contribution, respectively). `q_proj`,
`k_proj`, and `v_proj` read from the stream to produce queries, keys, and values consumed
inside the attention computation itself; `gate_proj` and `up_proj` read from the stream to
produce values consumed inside the MLP's gating. None of those five outputs are ever added
back to the stream directly, so ablating them wouldn't remove the direction from anything a
later layer actually sees.

**5.** `o_proj.weight`'s outputs are residual-stream vectors *produced by a computation*,
`y = W @ x`, for whatever `x` a given forward pass happens to feed it - so the direction has to
be projected out of every output that computation could ever produce. `embed_tokens.weight`'s
rows are not the output of any computation at all - row `i` already *is* the residual-stream
vector for token `i`, directly (index 0 of Lesson 01's `hidden_states`, before any decoder
layer runs). There's no `y = Wx` to factor through, so each row gets Lesson 07's projection
applied to it as its own vector - which is exactly what `ablate_matrix_input` does, one row at
a time, instead of projecting a matrix's outputs.

**6.** False. A general fine-tuning LoRA picks a rank `r` as a hyperparameter that
*approximates* some unknown, potentially full-rank ideal weight delta - a smaller `r` trades
away expressive power on purpose. Here the delta, `-d (d^T W)`, is an outer product of two
vectors, and an outer product is already exactly rank 1 by construction, for any two vectors -
there was never a higher-rank "ideal" delta being compressed down. `B @ A` reproduces `ΔW`
exactly (up to floating point), not approximately.

**7.** `merge_into_model` overwrites the base weights in place, so trying a different `(layer,
weight)` combination means reloading a fresh, unedited copy of the model from disk before each
trial - the entire point of a 200-trial search is to move fast, and a full reload per trial
defeats that. A rank-1 LoRA is a small delta kept separate from the base weights; attaching it
for one trial and detaching it before the next leaves the base weights untouched the whole
time, so the model is loaded exactly once for the entire search instead of once per candidate.
