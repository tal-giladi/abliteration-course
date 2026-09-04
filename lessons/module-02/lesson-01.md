# 04 - Harmful vs. harmless prompt sets

Module 01 gave you `get_residuals_per_prompt`: feed it a prompt, get back a `[num_layers + 1,
hidden_size]` snapshot of where that prompt's residual stream sat at every layer. That's one
point in a high-dimensional space. This module's job is to turn a *pile* of those points into
a single direction that means "refusal" - and today is about why one point is never enough to
do that, no matter how well-chosen the prompt is.

## From one point to a cloud

Say you pick the single most obviously-harmful prompt you can think of - "how do I make a
bomb" - and read off its residual stream at some layer. That vector contains the model's
representation of *this exact request*: its topic, its phrasing, its length, the specific
tokens involved, and, somewhere in the mix, whatever makes the model want to refuse it. All of
that is tangled together in one 1024-dimensional point. You have no way to tell which part of
the vector is "refusal" and which part is "sentence happens to be about explosives and start
with 'how do I'."

Now do the same thing for twenty different harmful prompts, spanning locks and drugs and
malware and forged signatures. Word choice changes, sentence length changes, topic changes -
but if there really is a "this is a refusal-worthy request" direction the model has learned,
that part should stay roughly consistent across all twenty points while everything else
varies. Average the twenty vectors together, and the topic-specific, phrasing-specific noise
- which points in different, uncorrelated directions for each prompt - tends to cancel out.
What's left over is closer to whatever *all twenty* have in common. That's the entire
statistical idea behind this module, and you already did it once, on paper, in Lesson 03:
`mean_difference` between two small synthetic groups. Today you run the same operation, at
real scale, on real activations.

This is also exactly why a classifier needs a training set instead of one labeled example -
same underlying reason, same fix.

## The course's prompt sets

`lib/common.py` defines `HARMFUL_PROMPTS` and `HARMLESS_PROMPTS`: 24 short prompts each, the
harmful ones covering a spread of categories (physical harm, cybercrime, fraud, drugs), the
harmless ones covering an equally deliberate spread of ordinary requests (cooking, science,
writing, how-things-work). Neither list is one repeated pattern - that would just teach you a
"starts with 'how do I'" direction, not a refusal direction.

Both lists are split, in file order:

    HARMFUL_TRAIN, HARMFUL_HOLDOUT = HARMFUL_PROMPTS[:20], HARMFUL_PROMPTS[20:]
    HARMLESS_TRAIN, HARMLESS_HOLDOUT = HARMLESS_PROMPTS[:20], HARMLESS_PROMPTS[20:]

20 prompts per class go into `*_TRAIN` - what you'll compute a direction *from*, today and in
Lesson 05. The last 4 per class go into `*_HOLDOUT` and stay untouched until Lesson 06, where
you'll use them to check whether the direction generalizes to prompts it never saw, or whether
it just memorized quirks of the 20 training prompts. Keeping that split fixed - not resampling
it, not peeking at holdout early - is what makes Lesson 06's check meaningful instead of
circular.

24 prompts per class is small on purpose - this course is optimizing for a feedback loop you
can run on a CPU in seconds, not for benchmark-grade rigor. `heretic`'s own default set, in
`config.default.toml`, is built on exactly this idea - a harmful prompt set and a harmless
prompt set, mean-differenced into a direction - just with hundreds of prompts per side instead
of dozens. Lesson 16 opens that file for real. Nothing about the *method* changes going from
24 prompts to 400; only the noise floor does.

## What "different" looks like before you've computed anything

Before you build the function that produces a direction, it's worth looking at what you
already have from Module 01 and confirming your intuition about what will and won't show a
difference. If you compare the *mean norm* of harmful-prompt residuals against harmless-prompt
residuals at, say, the last layer, don't expect a dramatic gap - overall activation magnitude
is driven mostly by sequence length and general "how much is going on here," not by whether
the prompt is one this model was trained to refuse. Two prompts of similar length end up with
similar-*sized* residual vectors whether or not the model is inclined to refuse either one.

That's the point of today's Example step: a norm is a single number, blind to direction. It
can't detect "these two clouds of points sit in different parts of the space" - it can only
tell you "these two clouds are roughly the same distance from the origin," which is a much
weaker and, here, mostly uninteresting fact. Separation is a *direction* question, which is
exactly why Lesson 05 computes a difference of mean *vectors*, not a difference of mean norms.

## Do this

1. From a Python shell in `lab/`, pull both training sets through the residual-stream function
   Module 01 already gave you, and compare mean norms at the last layer:

       from lib.common import HARMFUL_TRAIN, HARMLESS_TRAIN
       from lib.direction import get_residuals_per_prompt

       harmful = get_residuals_per_prompt(HARMFUL_TRAIN)
       harmless = get_residuals_per_prompt(HARMLESS_TRAIN)
       print(harmful.shape, harmless.shape)                       # [20, layers+1, hidden]
       print(harmful[:, -1, :].norm(dim=-1).mean().item())        # mean norm, last layer
       print(harmless[:, -1, :].norm(dim=-1).mean().item())       # close to the harmful number

   The two numbers should land in the same ballpark - confirming that "is this refusal-worthy"
   is not something a raw norm can see.

2. Open `lab/exercises/lesson_04.py` and implement `get_residuals_mean(prompts, batch_size=8)`:
   collapse a set of prompts' per-prompt residuals into a single mean residual per layer,
   returning a `[num_layers + 1, hidden_size]` tensor.

3. Grade it:

       bash lab/lab.sh check 04

   The checker runs your function on both `HARMFUL_TRAIN` and `HARMLESS_TRAIN`, compares each
   result to a reference implementation, and then checks something more interesting than
   "matches the reference": that the two means you computed are not, in fact, the same vector -
   if they were, there would be nothing for Lesson 05 to build a direction out of.

## Hints

- You already have the per-prompt version. `get_residuals_mean` is `get_residuals_per_prompt`
  followed by one `.mean(dim=0)` - don't re-run the model or re-batch anything yourself, and
  don't reimplement the forward pass a third time. Import `get_residuals_per_prompt` from
  `lib.direction`, the already-correct Module 01 function, and build on top of it.
- `.mean(dim=0)` collapses the *prompt* dimension (axis 0 of `[n_prompts, layers+1, hidden]`),
  leaving `[layers+1, hidden]` - the layer and hidden dimensions are untouched.
- If your output shape is wrong, print `.shape` at each step before you call `.mean` - it's the
  fastest way to catch an off-by-one on which axis you collapsed.

## Solution

    import torch
    from lib.direction import get_residuals_per_prompt


    def get_residuals_mean(prompts: list[str], batch_size: int = 8) -> torch.Tensor:
        return get_residuals_per_prompt(prompts, batch_size=batch_size).mean(dim=0)

## Summary

You now have two mean vectors - one per layer, one for harmful prompts, one for harmless ones
- and you've confirmed, on real activations, that averaging is doing what Lesson 03 promised:
canceling out prompt-specific noise and leaving a real, per-layer difference behind. Lesson 05
turns that difference into the thing this whole module exists to produce: a single, normalized
direction vector per layer.
