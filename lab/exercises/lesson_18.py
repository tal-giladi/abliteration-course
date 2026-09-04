# Lesson 18 - Making a real, scoped change
# Read lessons/module-06/lesson-03.md before filling this in.
#
# The actual code change goes inside lab/vendor/heretic (cloned in Lesson 15
# of Module 05), NOT in this file. This file just tells the checker which of
# the two shapes of change you picked and where to find it, so it can verify
# your change is actually live -- by importing it, or by running a real
# `heretic --help` against it.


def describe_change() -> dict:
    """Return a dict describing the scoped change you made to lab/vendor/heretic.

    Must have key "option" set to one of "scorer" or "setting".

    For option "scorer" (you added a new file under
    lab/vendor/heretic/src/heretic/scorers/):
        {
            "option": "scorer",
            "module": "heretic.scorers.<your_module>",   # e.g. "heretic.scorers.response_length"
            "class_name": "<YourScorerClass>",            # e.g. "ResponseLength"
        }

    For option "setting" (a new config default AND a new CLI flag are the
    SAME change in this codebase -- see the lesson for why -- so both use
    this option; you added a field to Settings in
    lab/vendor/heretic/src/heretic/config.py, and its default to
    lab/vendor/heretic/config.default.toml):
        {
            "option": "setting",
            "field_name": "<your_new_settings_field>",    # the Python field name, snake_case
        }
    """
    # TODO: implement your scoped change inside lab/vendor/heretic first,
    # then fill in and return one of the two description dicts above.
    raise NotImplementedError
