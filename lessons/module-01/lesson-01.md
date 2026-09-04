# 01 - Tokens, embeddings, and the residual stream

You've called LLM APIs. You send text in, text comes out, and everything in between has
probably felt like a black box worth not opening. Here's the box, in one paragraph, and then
we open it for real.

A transformer turns your prompt into tokens, turns each token into a vector (an
*embedding*), and then passes those vectors through a stack of identical-shaped layers. The
part almost every explanation gets slightly wrong: it is tempting to think of each layer as
*transforming* the vector into something new, the way a function transforms its input. That
is not what happens. Every layer **reads** the current vector and **adds something to it**.
Nothing is overwritten. Nothing is discarded. The vector that started as "the embedding of
this token" just keeps accumulating contributions, layer after layer, until the last layer's
output is read off to predict the next token.

This accumulating vector has a name: the **residual stream**. Everything in this course - the
whole idea of "abliteration" - is about that one object.

## Why it's called "residual"

Each layer computes something (call it `f(x)`) and the layer's actual output is
`x + f(x)`, not `f(x)` alone. `x` passes through unchanged, in parallel with whatever `f`
computes - a **residual connection**. Stack enough of these and the stream at layer *N* is
just the embedding, plus everything every earlier layer decided to add:

    stream_N = embedding + layer_0(stream_0) + layer_1(stream_1) + ... + layer_{N-1}(stream_{N-1})

That's the whole architecture, structurally. A transformer layer has two things that write to
the stream - **attention** (mixes information *between* token positions) and an **MLP**
(processes each position's vector on its own) - and both write the same way: read the
current stream, compute something, add it back.

## Why this matters for what we're about to do

If a layer only ever *adds* to the stream, then any concept the model has learned - "this
prompt is asking for something the model was trained to refuse" - has to be represented as
*something added to the stream that later layers can detect*. Not a separate signal, not a
special flag. A vector, added on top of everything else, sitting in the same
`hidden_size`-dimensional space as every other value in the stream. That single fact is what
makes this whole course possible: if refusal shows up as *a vector*, it can be found,
measured, and - because vector addition is invertible - **subtracted**.

That's three modules away. Today, you just need to see the stream with your own eyes.

## Do this

1. Set up the course's model once - this downloads and caches `Qwen/Qwen3-0.6B` (~1.2GB),
   which every remaining lesson reuses:

       bash lab/lab.sh up

2. Open a Python shell (`lab/.venv/bin/python` or `lab/.venv/Scripts/python.exe`) from the
   `lab/` directory and look around, without changing anything:

       from lib.common import load_model, load_tokenizer, encode_chat

       model = load_model()
       tok = load_tokenizer()
       print(model.config.hidden_size, model.config.num_hidden_layers)

       inputs = encode_chat(["What is the capital of France?"])
       out = model(**inputs, output_hidden_states=True)
       print(len(out.hidden_states))                 # one more than num_hidden_layers
       print(out.hidden_states[0].shape)              # [batch, seq, hidden_size]
       print(out.hidden_states[-1][:, -1, :].norm())   # the final stream, last token

3. `len(out.hidden_states)` is `num_hidden_layers + 1` - index 0 is the raw embedding output,
   *before* any layer has added anything; index `i` is the stream after layer `i - 1` has
   added its contribution. This indexing is the shape every exercise in this course uses, so
   sit with it for a second before moving on.

4. Now implement the exercise. Open `lab/exercises/lesson_01.py` and fill in
   `inspect_model(prompt)` - it should return a dict with the model's `hidden_size`,
   `num_hidden_layers`, and the L2 norm of the final-layer residual stream at the last token,
   for a single prompt. Everything you need is in the shell session above.

5. Grade it:

       bash lab/lab.sh check 01

## Hints

- `model.config.hidden_size` and `model.config.num_hidden_layers` - don't hardcode numbers,
  read them off the model so your function works for any model this course later swaps in.
- `encode_chat` takes a **list** of prompts, even for one. `encode_chat([prompt])`.
- `.norm()` on a tensor gives the L2 norm; call `.item()` to get a plain Python float back.
- The last token is always at index `-1` along the sequence dimension - this is guaranteed by
  left-padding, which `encode_chat` already sets up for you.

## Solution

    from lib.common import load_model, encode_chat

    def inspect_model(prompt: str) -> dict:
        model = load_model()
        inputs = encode_chat([prompt])
        out = model(**inputs, output_hidden_states=True)
        final_stream = out.hidden_states[-1][:, -1, :]
        return {
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "final_norm": final_stream.norm().item(),
        }

## Summary

A transformer doesn't transform your input through a pipeline - it keeps adding to one
vector, and every layer downstream reads whatever has accumulated so far. That vector is the
residual stream, and next lesson you'll capture it at every layer at once, not just the last
one.
