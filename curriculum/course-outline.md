# Course outline

**Abliteration: erasing a behavior from an LLM's weights** - 7 modules, 20 lessons, 20 graded
exercises, one capstone. Roughly 12-18 hours if you actually do the exercises, which is the
only way this course works.

## Audience and prerequisites

See [`brief/course-brief.md`](../brief/course-brief.md). In short: you're a software
developer, you know basic LLM/agent concepts, you've never opened up a transformer's
internals or touched PyTorch.

Required on the machine: Python 3.10+, and a CPU is enough for every exercise except the
real `heretic` run in Module 06. Setup is one command - `bash lab/lab.sh up` - which creates
a venv, installs `torch`/`transformers`, and downloads `Qwen/Qwen3-0.6B` once (~1.2GB,
cached for the rest of the course).

## Design principles

1. **Every lesson is graded against real tensors from a real model.** No lesson is "done"
   because you read it. `bash lab/lab.sh check NN` runs your code and asserts a measurable
   outcome - not "does it run", but "is the projection actually orthogonal", "does the
   direction actually separate held-out prompts", "did the refusal rate actually drop".
2. **You build the library, then you read the real one.** Modules 01-05 have you implement
   `lab/lib/` yourself, piece by piece. Module 06 opens the real `heretic` source and maps
   every piece back to what you already built - so it reads as recognition, not a wall of
   unfamiliar code.
3. **One model throughout.** `Qwen/Qwen3-0.6B`, chosen because it's small enough for a CPU
   and part of the same family already used in this conversation's Colab run against
   `Qwen/Qwen3-8B`. Everything after Lesson 01 uses it.
4. **Concepts arrive when the math needs them**, not before. The residual stream comes
   before directions; directions before projection; projection before "why a LoRA instead of
   editing weights"; scoring before search - because search without a score to search over
   is motion without direction (the pun is unavoidable and the course uses it once, here).
5. **Coherence is graded as hard as refusal.** Every exercise from Module 05 onward that
   measures "did it stop refusing" also measures "did it stay the same model otherwise" -
   because optimizing only the first number is how you get a fluent model that will do
   anything and remembers nothing.

## Module map

| # | Module | Lessons | You can, afterwards |
|---|---|---|---|
| 01 | Inside the residual stream | 01-03 | Hook a real model, read its hidden states, explain why a behavior is a direction |
| 02 | Finding the refusal direction | 04-06 | Extract a direction from real activations and prove it generalizes |
| 03 | The ablation trick | 07-09 | Derive the projection math and explain why Heretic uses a rank-1 LoRA, not a weight edit |
| 04 | Building the pipeline | 10-12 | Ablate a real model live, merge it permanently, and control per-layer strength |
| 05 | Making it automatic | 13-15 | Score refusal and coherence together, and search for parameters instead of guessing |
| 06 | Running the real thing | 16-18 | Run and configure `heretic` for real, read its actual source, and make a scoped change to it |
| 07 | Capstone and responsible use | 19-20 | Evaluate a decensored model properly, and build the whole pipeline from scratch |

## Lesson map

**Module 01 - Inside the residual stream**
1. Tokens, embeddings, and the residual stream - the mental model, hooking a real model for the first time
2. Reading hidden states across layers - `output_hidden_states`, what a "layer" writes
3. Behavior lives in a direction, not a neuron - the linear representation hypothesis, pure vector math

**Module 02 - Finding the refusal direction**
4. Harmful vs. harmless prompt sets - why the direction is a difference of means
5. Computing the direction - mean-difference, normalization, one vector per layer
6. Sanity-checking a direction - projecting held-out prompts and proving separation

**Module 03 - The ablation trick**
7. Orthogonal projection - the linear algebra of removing a direction from a vector
8. From vectors to weight matrices - which matrices write to the residual stream, and why those
9. Why a rank-1 LoRA instead of a weight edit - reversibility, and the factorization that makes it possible

**Module 04 - Building the pipeline**
10. Hooking every layer at once - live ablation during generation, no weights touched yet
11. Making the edit permanent - merging into the model's weights, saving and reloading
12. Choosing which layers and how strongly - per-layer ablation strength, and why not every layer gets the same

**Module 05 - Making it automatic**
13. Scoring a model - refusal detection and a coherence check, together
14. Search instead of guesswork - a random-search loop over ablation parameters
15. Reading Heretic's real optimizer - TPE vs. what you just built, mapped line by line

**Module 06 - Running the real thing**
16. Installing and configuring Heretic - `config.default.toml`, quantization, the Colab run revisited
17. Reading `model.py::abliterate()` for real - the production version of Module 03-04's toy code
18. Making a real, scoped change - a new scorer or config default, verified by actually running it

**Module 07 - Capstone and responsible use**
19. Evaluating a decensored model properly - refusal rate, coherence, and a capability spot-check together
20. Capstone: `mini-heretic.py` end to end, and where this technique stops being research

## Assessment

- **20 graded exercises**, one per lesson, run with `bash lab/lab.sh check NN`.
- **7 quizzes**, one per module, in [`assessments/`](../assessments/) - written to test
  diagnosis and judgement, not recall. Full answer keys.
- **The capstone** (Lesson 20) is the exam: a from-scratch script that computes a direction,
  ablates a real model, runs a parameter search, and produces a measured before/after report
  - graded on the numbers it produces, not on matching a reference implementation exactly.

## What this course does not cover

Full fine-tuning or RLHF, quantization internals (used as a black box in Module 06, not
derived), any jailbreak technique other than directional ablation, and multimodal models
(Heretic supports them; this course's model is text-only throughout). Lesson 17 gives you
the model heretic's production code is built on, in enough depth to read the rest of the
repository yourself.
