# Lesson 20 - Capstone: mini-heretic.py
# Read lessons/module-07/lesson-02.md before filling this in.

from lib import ablate, common, direction, score


def run_mini_heretic(model_name: str = common.MODEL_ID) -> dict:
    """Assemble everything from Modules 01-05's lab/lib into one standalone
    pipeline: compute a direction from lib.common.HARMFUL_TRAIN /
    HARMLESS_TRAIN, merge it permanently into a loaded model with
    lib.ablate.merge_into_model, and return a
    lib.score.full_report(...)-shaped dict — no calls into
    lab/vendor/heretic (Module 06), only lab/lib.

    model_name defaults to the one model this lab has cached
    (lib.common.MODEL_ID). If the model isn't cached, raise a RuntimeError
    with a clear message to run `bash lab/lab.sh up` first, instead of
    letting a raw OSError from transformers leak out.
    """
    # TODO:
    # 1. If model_name isn't lib.common.MODEL_ID, this lab has nothing else
    #    cached — raise ValueError saying so.
    # 2. model = lib.common.load_model(); catch OSError (raised when nothing
    #    is cached and there's no network) and re-raise a RuntimeError with a
    #    "run bash lab/lab.sh up first" message.
    # 3. Capture "before" logits on the 5 capability-question prompts
    #    (lib.score.CAPABILITY_QUESTIONS) BEFORE any ablation — this is your
    #    only chance, since merge_into_model edits weights in place
    #    (Lesson 11) and there's no unablated copy left afterward.
    # 4. Compute the direction: lib.direction.get_residuals_mean on
    #    HARMFUL_TRAIN and HARMLESS_TRAIN, then lib.direction.compute_direction
    #    on the two means.
    # 5. Merge it into the model: lib.ablate.merge_into_model(model,
    #    directions, weights). A fixed strength (e.g. 1.0 at every layer) is
    #    fine here; a small lib.search.random_search sweep over strength is a
    #    documented alternative — see the lesson for the tradeoff.
    # 6. Generate + score: lib.score.refusal_rate on HARMFUL_HOLDOUT,
    #    lib.score.capability_spot_check on the 5 capability prompts,
    #    lib.score.coherence_score comparing fresh logits on those same 5
    #    prompts against the "before" logits from step 3.
    # 7. Return lib.score.full_report(refusal, coherence, capability).
    raise NotImplementedError
