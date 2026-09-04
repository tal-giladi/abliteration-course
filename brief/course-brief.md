# Course brief

## The learner

One specific person: a software developer with basic LLM and agent knowledge - they've
called model APIs, written prompts, maybe built a RAG pipeline or an agent loop - but never
opened up a transformer's internals, never touched PyTorch in anger, and has no research-ML
background. They asked how to run
[`p-e-w/heretic`](https://github.com/p-e-w/heretic) in Colab, ran it successfully, then
asked whether they could actually understand and modify the codebase. This course is the
answer.

Assumed: Python, basic linear algebra (a dot product, a vector norm - nothing past that),
comfort calling an HTTP API or a library you didn't write. Not assumed: PyTorch, any prior
exposure to transformer internals, any ML research background, any linear algebra beyond
high-school vectors.

## The outcome

By the end, they can explain - correctly, not just by analogy - what "abliteration" does to
a language model: how a behavior like refusal is represented as a direction in the residual
stream, how that direction is extracted from real activations, and how it is projected out
of a model's weights so the behavior disappears without retraining. They will have built
every piece of that pipeline themselves in Python against a real model (`Qwen/Qwen3-0.6B`,
small enough to run on a CPU), not just read about it.

They can then open `heretic`'s actual source (`model.py`, `main.py`, `evaluator.py`,
`config.py`, `scorers/`) and recognize their own toy implementation in it - and make a real,
scoped change to it (a new scorer, a new config default, a new CLI flag) and verify the
change works by running it.

They also come away able to state, in their own words, what this technique is legitimately
for (safety research, red-teaming a deployment's own guardrails, portability testing) and
where using it crosses into something they shouldn't do - because a course that teaches you
to remove a safety behavior without ever discussing why that behavior existed is teaching
half the material.

## The shape

**Interactive, not a book.** Twenty graded exercises, each checked by a script that runs
your code against real tensors from a real model and asserts a real, measurable outcome
(`bash lab/lab.sh check NN`) - a projection that's actually orthogonal, a direction that
actually separates held-out prompts, a refusal rate that actually drops. A lesson is not
complete because it was read.

**One real model throughout, chosen for the CPU.** `Qwen/Qwen3-0.6B` - part of the same
model family the learner already ran through `heretic` in Colab - is small enough to load
and run forward/backward passes on a laptop CPU in seconds, so every exercise from Module 01
onward runs against real activations from a real instruction-tuned model, not a toy. No GPU,
no Colab, and no `bitsandbytes` are required until Module 06, where the learner runs the
real `heretic` CLI end to end (locally against the 0.6B model, and optionally in Colab
against a larger one, exactly as covered earlier in this conversation).

**Concept before code, always against the real thing.** Every mechanism - the residual
stream, the direction, the projection, the LoRA trick, the scoring, the search - is
explained in plain language first, then implemented by the learner in a shared library
(`lab/lib/`) they build up module by module, checked against reference tensors.

**Nothing hand-waved, including the parts a shortcut course would skip.** Why LoRA instead
of editing weights directly. Why the direction is a *difference of means*, not a single
prompt's activation. Why coherence has to be scored alongside refusal, or "success" is a
model that agrees with everything and remembers nothing. Why heretic uses TPE search instead
of a grid. The real `heretic` source is read line-by-line in Module 06, not summarized.

## Constraints the design had to respect

- **CPU-only for every exercise except the real `heretic` run.** The learner may not have a
  GPU on their development machine even though they have Colab access - the exercises must
  work without either.
- **One model download, cached once.** `bash lab/lab.sh up` downloads `Qwen/Qwen3-0.6B`
  (~1.2GB) a single time; every later lesson reuses the cache.
- **Every exercise is graded against real model output**, not mocked tensors, from Module 02
  onward - Module 01's last lesson is the only place small synthetic vectors are used, and
  only because the point of that one lesson is the pure math, before a real model is
  involved at all.
- **Windows 11 + Git Bash**, matching the environment this learner already works in; `lab.sh`
  is a Bash script that shells out to a Python venv, the same split used by this course
  creator's other lab-based courses.

## Non-goals

Training a model from scratch, quantization internals, any technique other than directional
ablation (no full fine-tuning, no RLHF, no other jailbreak methods), and a survey of every
paper in the space. This course teaches one technique, end to end, deeply enough that the
learner can read and extend the one real tool that implements it.
