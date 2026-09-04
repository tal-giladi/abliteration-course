# Lesson 15 - Reading Heretic's real optimizer
# Read lessons/module-05/lesson-03.md before filling this in.
#
# Clone the real repository first, if you haven't:
#     git clone https://github.com/p-e-w/heretic lab/vendor/heretic
#
# Answer each question below by citing exactly where it happens in the real
# heretic source, as a string "path/to/file.py::name" — the file path
# relative to lab/vendor/heretic/src/heretic/, and the exact function or
# class name where the answer actually lives. Go verify each one against the
# cloned files yourself; the checker checks the real files, not the lesson.

# TODO: replace every "???" below with a real citation from the cloned repo.
ANSWERS = {
    # Where is the Optuna study created (optuna.create_study(...), with a
    # TPESampler)?
    "where_is_the_optuna_study_created": "???",
    # Where is the search's objective function — the one called once per
    # trial, that resets the model, abliterates it, and scores it — defined?
    "where_is_the_search_objective_function_defined": "???",
    # Where does a trial actually get scored — i.e. where do all the
    # configured scorer plugins get asked for their Score on the current
    # model?
    "where_does_a_trial_actually_get_scored": "???",
    # Which scorer class is the real-world equivalent of this course's
    # refusal_rate — a keyword/marker match over model responses?
    "which_scorer_class_is_the_refusal_rate_equivalent": "???",
    # Which scorer class is the real-world equivalent of this course's
    # coherence_score — a KL divergence between the current and baseline
    # next-token distributions?
    "which_scorer_class_is_the_coherence_equivalent": "???",
}
