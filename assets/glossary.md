# Glossary

**Residual stream** - the single vector, per token position, that every transformer layer
reads from and adds to. Nothing overwrites it; it accumulates from the embedding layer
through the final layer. See Lesson 01.

**Hidden states** - the residual stream's value at each layer boundary, returned by
`output_hidden_states=True` as a tuple of `num_layers + 1` tensors. See Lesson 02.

**Linear representation hypothesis** - the (empirically supported) idea that a concept a
model has learned is represented as a direction in activation space, detected via a dot
product, rather than a single neuron. See Lesson 03.

**Direction** - a unit vector, one per layer, that separates two groups of activations (e.g.
harmful vs. harmless prompts). Computed as a normalized difference of means. See Lesson 05.

**Projection scores / separation** - how far a prompt's residual stream sits along a
direction; used to verify a direction actually detects what you think it detects, on
held-out data. See Lesson 06.

**Orthogonal projection / ablation** - removing a direction's component from a vector:
`v' = v - (v . d) d`. The core operation behind "abliteration." See Lesson 07.

**Abliteration** - directional ablation: projecting a behavior's direction out of the weight
matrices that write to the residual stream, so the model can no longer represent that
behavior. The technique implemented by `heretic`.

**LoRA (Low-Rank Adaptation)** - expressing a weight change as a low-rank product `B @ A`
instead of editing the base weight directly. A rank-1 LoRA can express an exact orthogonal
projection. See Lesson 09.

**Forward hook** - a function PyTorch calls with a module's output right after it runs,
letting you inspect or modify it live without changing any weights. See Lesson 10.

**Merge** - applying a live hook's edit permanently into a model's actual weight tensors, so
no hook is needed afterward. See Lesson 11.

**Refusal rate** - the fraction of model responses to a prompt set that look like a refusal.
Lower generally means "more decensored." See Lesson 13.

**Coherence score** - how close an edited model's output distribution stays to the original
model's, on the same prompts. Prevents mistaking "broken" for "decensored." See Lesson 13.

**TPE (Tree-structured Parzen Estimator)** - the search strategy `heretic`'s real optimizer
uses: unlike random search, it models which regions of parameter space scored well and
samples more from there. See Lesson 15.

**Capability spot-check** - a small, fixed set of general-knowledge questions used to catch
the failure mode where a model "stops refusing" because it has stopped being coherent at
all, not because it was successfully ablated. See Lesson 19.
