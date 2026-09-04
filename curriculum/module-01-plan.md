# Module 01 plan - Inside the residual stream

**Lessons 01-03. Prerequisite: Python, basic vectors. Produces: a working venv, a cached
model, and the mental model everything else builds on.**

## Objectives

1. Explain a transformer as a stack of layers that all read from and write to one shared
   vector (the residual stream), rather than a pipeline that transforms data stage by stage.
2. Extract and inspect real hidden states from a real model with `output_hidden_states=True`.
3. Explain why a behavior like refusal is represented as a *direction* in that stream rather
   than a single neuron, and do the vector arithmetic that claim rests on.

## Lessons

### 01 - Tokens, embeddings, and the residual stream
- **Concept**: tokenization -> embedding -> the residual stream as the model's shared
  notebook; every attention block and every MLP block reads the current stream and adds
  something back to it (a *residual* connection, hence the name); nothing overwrites it,
  everything accumulates onto it.
- **Example**: `bash lab/lab.sh up` downloads and caches `Qwen/Qwen3-0.6B`; then hook layer 0
  and print the residual stream's shape and norm for one real prompt.
- **Practice**: write a script that loads the model, runs one prompt, and reports
  `hidden_size`, `num_hidden_layers`, and the L2 norm of the final hidden state.
- **Summary**: a transformer doesn't transform your input through a pipeline - it keeps
  adding to one vector, and everything downstream reads whatever is in it so far.

### 02 - Reading hidden states across layers
- **Concept**: `output_hidden_states=True` returns one snapshot of the stream per layer
  boundary (`num_layers + 1` of them, including the raw embedding output); norm growth
  across layers; why the *last token's* position is what matters for a decoder-only model
  generating the next token.
- **Example**: capture hidden states for two very different prompts and compare their norm
  profiles layer by layer.
- **Practice**: implement `get_residuals_per_prompt` (this becomes `lab/lib/direction.py`,
  built here) and pass a shape/dtype check against real model output.
- **Summary**: every lesson from here on operates on this one object - a per-layer, per-token
  snapshot of the residual stream - so getting comfortable with its shape now pays for the
  rest of the course.

### 03 - Behavior lives in a direction, not a neuron
- **Concept**: the linear representation hypothesis in one sentence - a concept a model has
  learned is (approximately) a direction in activation space, and the model detects that
  concept by how much a vector points along that direction (a dot product); why "the refusal
  neuron" is the wrong mental model and "the refusal direction" is the right one; cosine
  similarity as "how aligned are two directions".
- **Example**: two small synthetic vectors standing in for "mean harmful activation" and
  "mean harmless activation" - the only lesson in the course that doesn't use the real model,
  because the point here is the math, cleanly, before a 1024-dimensional real one arrives.
- **Practice**: implement a cosine-similarity function and a difference-vector function by
  hand; checked against exact reference values.
- **Summary**: a direction is just a vector that means something because of what it's aligned
  with - and Module 02 is entirely about how you find the one that means "refuse this".

## Dependencies

01 -> 02 -> 03, strictly. Lesson 02's `get_residuals_per_prompt` is the function Module 02
calls on real prompts starting in Lesson 04.

## Misconceptions to hit head-on

- "The model transforms my input step by step, like a pipeline." (It keeps *adding* to one
  vector; nothing is discarded until the final read-out.)
- "There's a single 'refusal neuron' somewhere in the network." (No individual dimension of
  the residual stream reliably means "refuse" - a *direction*, a weighted combination of many
  dimensions, does.)
- "Cosine similarity and dot product are the same thing." (Dot product also depends on
  magnitude; cosine similarity is dot product after normalizing both vectors to length 1 -
  this distinction matters the moment normalization gets introduced in Lesson 05.)
