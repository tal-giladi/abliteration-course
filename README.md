> ⚠️ **Publishing warning:** sensitive topic - review before importing to the online academy. See [PUBLISHING_WARNING.md](PUBLISHING_WARNING.md).

# Abliteration: erasing a behavior from an LLM's weights

**7 modules · 20 lessons · 20 graded exercises · 7 quizzes · 1 capstone**

A hands-on course for software developers who want to actually understand - and be able to
modify - [`p-e-w/heretic`](https://github.com/p-e-w/heretic), the tool that automatically
removes refusal behavior from language models. You do not read this course - you run it.
Every lesson ends in an exercise **graded against real activations from a real model**
(`Qwen/Qwen3-0.6B`, small enough for a CPU), and by Module 06 you're reading and modifying
Heretic's actual production source.

---

## Start here

```bash
bash lab/lab.sh up          # create a venv, install torch/transformers, cache the model (~1.2GB, once)
```

Then open [`lessons/module-01/lesson-01.md`](lessons/module-01/lesson-01.md), do the exercise
at the bottom, and run:

```bash
bash lab/lab.sh check 01
```

## What you need

Python 3.10+ and a CPU. No GPU and no Colab are required for any of the 20 graded exercises -
only Module 06's optional larger-model run revisits the Colab command already used earlier
against `Qwen/Qwen3-8B`. Windows 11 + Git Bash is the environment this course was written for;
macOS and Linux need no changes.

| Command | What it does |
|---|---|
| `bash lab/lab.sh up` | create the venv, install requirements, cache the model |
| `bash lab/lab.sh check NN` | grade exercise NN against real model output |
| `bash lab/lab.sh hint NN` | point you at the lesson's Hints section |
| `bash lab/lab.sh solve NN` | copy the reference solution into your exercise file |
| `bash lab/lab.sh reset NN` | restore exercise NN to its original stub |
| `bash lab/lab.sh status` | which exercises currently pass |

---

## Contents

### Lessons - `lessons/module-NN/lesson-XX.md`

| Module | Lessons | Title |
|---|---|---|
| 01 | 3 | Inside the residual stream |
| 02 | 3 | Finding the refusal direction |
| 03 | 3 | The ablation trick |
| 04 | 3 | Building the pipeline |
| 05 | 3 | Making it automatic |
| 06 | 3 | Running the real thing |
| 07 | 2 | Capstone and responsible use |

Each lesson: the mechanism explained properly against a real model, a graded exercise, hints,
and a full solution.

### Curriculum - `curriculum/`
[`course-outline.md`](curriculum/course-outline.md) and seven module plans with objectives,
dependencies, and the misconceptions each module is written to break.

### Assessments - `assessments/`
Seven quizzes with full answer keys, weighted toward diagnosis and judgement rather than
recall.

### Lab - `lab/`
`lab.sh` (the grader), `lib/` (the reference library you build up module by module -
`common.py`, `direction.py`, `ablate.py`, `score.py`, `search.py`), `exercises/` (your stubs),
`checks/` (20 graders), `solutions/`.

### Assets - `assets/`
[`glossary.md`](assets/glossary.md).

---

## How it is built

**One real model throughout.** `Qwen/Qwen3-0.6B` - small enough to load and run on a laptop
CPU in seconds, part of the same model family already run through `heretic` in Colab earlier.
Every exercise from Module 02 onward runs against real activations from a real
instruction-tuned model, not a mock.

**You build the library, then you read the real one.** Modules 01-05 have you implement
`lab/lib/` yourself, one function per lesson, checked against a working reference
implementation. Module 06 opens Heretic's actual `model.py`, `main.py`, `evaluator.py` and
`config.py` and maps every piece back to what you already built - so it reads as recognition,
not a wall of unfamiliar code.

**Nothing hand-waved.** Why the direction is a difference of means, not one prompt's
activation. Why a rank-1 LoRA is exact, not approximate, for this specific edit. Why
coherence has to be scored alongside refusal, or "success" is a model that agrees with
everything and remembers nothing. The real `heretic` source is read line by line, not
summarized.

**Responsible by design, not by omission.** Lesson 20 closes the course with an explicit
section on what this technique is legitimately for and where using it crosses a line - a
course that teaches the mechanism without ever naming its misuse potential has taught half of
it.

---

## Cleaning up

```bash
rm -rf lab/.venv lab/.cache lab/vendor
```

Removes the Python venv, the cached model, and the cloned copy of `heretic` used in Modules
05-06. Nothing else on your machine is touched.
