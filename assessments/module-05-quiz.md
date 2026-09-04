# Module 05 quiz - Making it automatic

Seven questions. Answers and explanations at the bottom - try all seven first.

---

**1.** A teammate proposes: "just run the search, and pick whichever trial has the lowest
`refusal_rate`." What's wrong with that selection rule, and what would you check instead?

**2.** You ablate every layer at strength 1.0 (the maximum). `refusal_rate` on the held-out
harmful set drops to 0%. Is this a successful abliteration? What single additional number would
settle the question, and what result on that number would tell you it *isn't*?

**3.** `coherence_score(logits_a, logits_b)` returns `1.0` when you call it with the same tensor
for both arguments. Explain why, in terms of what the function actually computes - don't just say
"they're equal."

**4.** Two engineers both implement `random_search` correctly per this course's spec, with the
same `seed` and the same `param_space`. Their `best_score` results differ. Is one of them wrong?
Why or why not, and why does this course's Lesson 14 checker avoid comparing your output directly
against a reference run?

**5.** In Heretic's real `main.py`, `TPESampler` is constructed with `n_startup_trials=...`. What
would go wrong early in a search run if `n_startup_trials` were set to `0`?

**6.** "TPE is a fundamentally different algorithm from random search - they don't have much in
common." A colleague says this after skimming Optuna's documentation. Correct them in two or
three sentences, referencing the actual structure both algorithms share.

**7.** Heretic's `KLDivergence` scorer is configured with `optimization = "minimize"` in
`config.default.toml`, while this course's `coherence_score` treats a *higher* number as better.
Are these two functions measuring different things? Reconcile the two conventions.

---

## Answers

**1.** Selecting purely on `refusal_rate` rewards a model that agrees with everything, including a
model that's been damaged into incoherence and therefore can't form a refusal in the first place -
`refusal_rate` alone can't distinguish "the behavior was actually removed" from "the model is
broken and produces nothing refusal-shaped." Check `coherence_score` on the same trial before
trusting the refusal number; a genuine success needs low `refusal_rate` *and* `coherence_score`
still near 1.0.

**2.** Not necessarily - a 0% refusal rate is exactly what you'd also see from a model that's been
ablated into producing degraded, off-topic, or repetitive output that simply no longer resembles a
refusal (or anything else coherent). `coherence_score`, computed between the edited model's
logits and the original model's logits on the same (non-refusal-related) prompts, settles it: a
value collapsed well below 1.0 means the edit damaged the model generally, not just the targeted
behavior, even though the refusal number looks perfect.

**3.** `coherence_score` computes `exp(-KL(base || edited))` from the two logits tensors. When
both arguments are the same tensor, the KL divergence between a distribution and itself is exactly
0 by definition (there's no difference to measure), and `exp(-0) = 1.0`. It isn't `1.0` because
"the inputs are equal" as a special case in the code - it falls out of the KL divergence formula
itself hitting its minimum value at zero.

**4.** Neither is wrong. `random_search`'s spec (draw one uniform value per parameter, per trial,
`n_trials` times, track the best) doesn't pin down the exact internal order or mechanism of the
draws - a `for` loop building a dict one key at a time and a different-but-equivalent
implementation can call the underlying RNG in a different sequence and land on different actual
numbers, even with an identical `seed`, while both are fully correct per the spec. That's exactly
why the Lesson 14 checker grades *behaviorally* - it checks that your search finds a point near a
known optimum, not that your numbers match a specific reference run bit-for-bit; matching RNG
call order exactly would be grading an implementation detail the spec never promised.

**5.** With `n_startup_trials=0`, TPE would try to split trials into "good" and "not as good"
groups and bias sampling toward the good region before it has *any* trials to learn from - there's
no history yet to model. In practice, TPE implementations fall back to something like uniform
random sampling during this bootstrap phase specifically to avoid that problem; setting it to 0
removes the safety margin and risks the sampler making a confident-looking but data-free guess
about "good" regions from the very first trial, before it has any real evidence.

**6.** Both are the identical loop - propose parameters, evaluate them, keep track of what worked,
repeat - and neither one uses gradients or a closed-form solution to pick the next point. The only
real difference is the proposal step: `random_search` samples every trial uniformly at random with
no memory of past trials, while TPE models which regions scored well versus poorly and samples the
next trial from a distribution biased toward the "good" region. It's the same algorithm family
with one component swapped, not two unrelated techniques.

**7.** No - both measure the same underlying quantity, KL divergence between the current model's
and the original model's next-token distributions on the same prompts, computed with the same
`F.kl_div(..., log_target=True)` call. They differ only in which direction "better" points:
Heretic's `KLDivergence` scorer reports the raw KL value directly, where 0 is the ideal and larger
is worse, hence `optimization = "minimize"`; this course's `coherence_score` wraps that same KL
value in `exp(-KL)` specifically so it lands on a bounded 0-1 scale where *larger* is better,
which is the convenient shape for summing directly into Lesson 14's single blended objective
alongside `1 - refusal_rate`. Same measurement, two different framings chosen for two different
downstream uses.
