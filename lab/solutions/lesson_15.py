# Answer key based on the real p-e-w/heretic repository (src/heretic/), read
# for this lesson. Question wording may vary; these citations are the ones
# this lesson's markdown walks through and verifies against the real source.

ANSWERS = {
    "where_is_the_optuna_study_created": "main.py::run",
    "where_is_the_search_objective_function_defined": "main.py::objective",
    "where_does_a_trial_actually_get_scored": "evaluator.py::get_scores",
    "which_scorer_class_is_the_refusal_rate_equivalent": "scorers/keyword_rate.py::KeywordRate",
    "which_scorer_class_is_the_coherence_equivalent": "scorers/kl_divergence.py::KLDivergence",
}
