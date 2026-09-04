import os
import re
import subprocess
import sys
from pathlib import Path

from _lib import check, fail

from exercises.lesson_18 import describe_change

LAB_DIR = Path(__file__).resolve().parent.parent
VENDOR_DIR = LAB_DIR / "vendor" / "heretic"
VENDOR_SRC = VENDOR_DIR / "src"

if not (VENDOR_SRC / "heretic").is_dir():
    fail(
        "lab/vendor/heretic isn't cloned yet — clone it first (Lesson 15): "
        "git clone https://github.com/p-e-w/heretic lab/vendor/heretic"
    )

try:
    change = describe_change()
except NotImplementedError:
    fail("describe_change() is still a stub — fill in the TODO in exercises/lesson_18.py")

check(
    isinstance(change, dict) and change.get("option") in ("scorer", "setting"),
    "describe_change() returns option 'scorer' or 'setting'",
)

# Make lab/vendor/heretic/src take precedence over any pip-installed heretic-llm, so
# every `heretic.*` import below resolves to YOUR edited clone.
sys.path.insert(0, str(VENDOR_SRC))

if change["option"] == "scorer":
    module_name = change.get("module", "")
    class_name = change.get("class_name", "")
    check(bool(module_name) and bool(class_name), "module and class_name are set")

    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        fail(
            f"could not import {module_name!r} from lab/vendor/heretic/src ({error}). "
            "Check the file exists under lab/vendor/heretic/src/heretic/scorers/ "
            "and that heretic-llm's dependencies are installed (Lesson 16)."
        )

    cls = getattr(module, class_name, None)
    check(cls is not None, f"{module_name} defines {class_name!r}")

    try:
        from heretic.scorer import Scorer
    except Exception as error:
        fail(f"could not import heretic.scorer.Scorer to validate against ({error})")

    check(isinstance(cls, type) and issubclass(cls, Scorer), f"{class_name} subclasses heretic.scorer.Scorer")

    try:
        cls.validate_contract()
    except Exception as error:
        fail(f"{class_name} fails heretic's own plugin contract check: {error}")

    check(hasattr(cls, "get_score") and callable(getattr(cls, "get_score")), f"{class_name} implements get_score()")

    print("Lesson 18: all checks passed.")

elif change["option"] == "setting":
    field_name = change.get("field_name", "")
    check(bool(field_name) and field_name.isidentifier(), "field_name is set and a valid Python identifier")

    flag = "--" + field_name.replace("_", "-")

    config_toml_path = VENDOR_DIR / "config.default.toml"
    check(config_toml_path.is_file(), "lab/vendor/heretic/config.default.toml exists")
    config_toml = config_toml_path.read_text(encoding="utf-8")
    check(
        re.search(rf"^\s*{re.escape(field_name)}\s*=", config_toml, re.MULTILINE) is not None,
        f"{field_name!r} has a documented default in config.default.toml",
    )

    import importlib

    try:
        config_module = importlib.import_module("heretic.config")
    except Exception as error:
        fail(f"could not import heretic.config from lab/vendor/heretic/src ({error})")

    check(
        field_name in config_module.Settings.model_fields,
        f"heretic.config.Settings declares a {field_name!r} field",
    )

    env = {**os.environ, "PYTHONPATH": str(VENDOR_SRC)}
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.argv = ['heretic', '--help']; import heretic.main"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except Exception as error:
        fail(f"running `heretic --help` failed: {error}")

    help_text = result.stdout + result.stderr
    check(bool(help_text.strip()), "heretic --help produced output")
    check(flag in help_text, f"heretic --help lists {flag!r}")

    print("Lesson 18: all checks passed.")
