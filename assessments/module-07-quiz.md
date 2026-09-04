# Module 07 quiz - Capstone and responsible use

Seven questions. Answers and explanations at the bottom - try all seven first.

---

**1.** A colleague says "I don't need `capability_spot_check` - `refusal_rate` going down and
`coherence_score` staying reasonable already tells me the ablation worked." Explain, in one or
two sentences, the failure mode this misses.

**2.** Another colleague says "`capability_spot_check` is basically a small benchmark, so a
model that passes it is a good model." Correct them - what is `capability_spot_check` actually
for, and what can't it tell you?

**3.** `run_full_report(model=None, base_logits=None)` from Lesson 19 takes `base_logits` as an
*optional* argument instead of just computing coherence itself every time. Why can't it always
compute coherence on its own?

**4.** In `mini-heretic.py` (Lesson 20), `base_logits` is captured *before* `merge_into_model`
runs, not after. What would `coherence_score` report if you captured it after the merge
instead, and why would that number be real but meaningless?

**5.** The Module 07 plan says Lesson 20 "imports from `lab/lib` exclusively, not from
`lab/vendor/heretic`." What does running the capstone this way prove that running the real
`heretic` CLI in Module 06 didn't?

**6.** Name the three legitimate uses of abliteration this course names in Lesson 20's closing
section.

**7.** Using this course's own prompt sets as the concrete example, describe the difference
between using an ablated model to audit your own product's safety layer, and using one to
generate the content in `HARMFUL_PROMPTS` for real. What's the one-sentence distinction the
course ends on?

---

## Answers

**1.** `coherence_score` compares next-token logits at a single forward pass, on one prompt at
a time - it can look locally fine even while a model's *generated* output degrades over many
tokens (repetition, drift, incoherence). A model can score a dropping refusal rate and an
unremarkable coherence score while actually just being too broken to refuse - `capability_spot_check`
catches that by testing something coherence never looks at: does the model still produce
correct answers to trivial questions.

**2.** It's a five-question smoke test for "did this ablation break the model," not a
capability evaluation - it can't tell you the model got smarter, dumber, or better at
anything specific, only whether it still knows basic facts it should trivially get right. A
model passing it is not thereby shown to be "good" in any general sense - only shown to not be
obviously broken.

**3.** Coherence needs a "before" copy of the unablated model's logits to compare against, and
by the time `run_full_report` is called, the model it's evaluating may already be an ablated
one with no unablated version left in memory - `merge_into_model` edits weights in place
(Lesson 11), so there's nothing to reload without a saved copy. `run_full_report` can't
conjure that copy after the fact, so it accepts it as an argument from whoever captured it
earlier, before ablation happened.

**4.** It would report something close to a perfect 1.0 - the ablated model's logits compared
against a copy of the ablated model's own logits are nearly identical by construction. The
number would be entirely real (no bug, no error) and would tell you nothing, because
"coherence" is supposed to mean "close to the *original* model," and by that point the
original model no longer exists anywhere in the comparison.

**5.** It proves the toy library built across Modules 01-05 actually works standalone end to
end - that the pipeline isn't secretly leaning on the real `heretic` package somewhere, and
that everything learned by building it by hand transfers to a script with no dependency on
the production tool at all. Module 06 proved the *real* tool works and that you can read and
change its source; Lesson 20 proves your own version, built from first principles, does the
same job on its own.

**6.** Safety research (understanding whether alignment techniques are robust or just
cosmetic), red-teaming a deployment's own guardrails before an attacker does, and testing a
behavior's portability across providers or hosts.

**7.** Auditing your own product means running the technique against a model you control and
reading the report - an experiment. Generating the actual content in `HARMFUL_PROMPTS` - real
phishing emails, real working ransomware, real instructions - with the intent of using it for
real, is the thing the base model's safety training existed to prevent, done anyway. The
course's closing line: "I can" and "I should" are different questions, and passing the
capstone's check only ever answered the first one.
