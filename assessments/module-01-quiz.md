# Module 01 quiz - Inside the residual stream

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** A colleague says "layer 5 transforms the output of layer 4 into something new."
Correct them in one or two sentences, using the term "residual connection."

**2.** `output_hidden_states=True` on a model with 28 decoder layers returns a tuple of how
many tensors? What does entry `0` represent, and what does entry `28` represent?

**3.** Why does this course slice the residual stream at the *last* token position
(`[:, -1, :]`) instead of, say, the first token or an average over all tokens?

**4.** `encode_chat` left-pads its input. Explain what would go wrong with `[:, -1, :]` if it
right-padded instead, for a batch containing one short prompt and one long prompt.

**5.** Two vectors `a` and `b` point in exactly the same direction, but `b` is five times
longer than `a`. What is `cosine_similarity(a, b)`? What is `a @ b` (the raw dot product) -
larger than, smaller than, or equal to `a @ a`?

**6.** Why does this course compute a *mean* difference over many prompts (Lesson 03's
`mean_difference`) instead of using the activation from a single, very clear example prompt?

---

## Answers

**1.** Layer 5 does not replace layer 4's output. It computes something and *adds* it to
layer 4's output - the residual connection passes the input through unchanged in parallel
with whatever the layer computes, so the stream accumulates rather than getting replaced.

**2.** 29 tensors. Entry `0` is the raw embedding output, before any decoder layer has run.
Entry `28` is the residual stream after all 28 layers have added their contributions - the
one actually used to predict the next token.

**3.** A decoder-only model predicts the next token from whatever is in the stream at the
current last position - that is the only position whose final-layer state is used for the
prediction. Earlier positions matter for how attention built up the last position's stream,
but their own final-layer states aren't what the model reads to decide what comes next.

**4.** With right-padding, `[:, -1, :]` would grab a pad token's residual stream for the
shorter prompt in the batch, not its last real token - silently wrong output with no error
raised. Left-padding guarantees index `-1` is always the last real token, for every sequence
in the batch, regardless of length.

**5.** `cosine_similarity(a, b) = 1.0` - cosine similarity is magnitude-independent, and same
direction means the angle between them is zero regardless of length. The raw dot product
`a @ b` is *larger* than `a @ a`, because `b`'s extra length multiplies straight through the
dot product even though the direction didn't change - which is exactly why raw dot products
aren't used to compare two arbitrary directions.

**6.** A single prompt's activation carries noise specific to that one prompt - word choice,
sentence length, topic details that have nothing to do with "is this a refusal-worthy
request." Averaging over many prompts cancels out what's specific to each one and leaves
(approximately) only what's common across the whole group - the same reason you wouldn't
train a classifier on a single labeled example.
