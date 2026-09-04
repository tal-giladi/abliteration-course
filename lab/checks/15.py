import sys

from _lib import LAB_DIR, check, fail

VENDOR_DIR = LAB_DIR / "vendor" / "heretic" / "src" / "heretic"

if not VENDOR_DIR.is_dir():
    print(
        "lab/vendor/heretic hasn't been cloned yet — Lesson 15 needs the real "
        "source to check your answers against.\n"
        "Run this first, then re-run the check:\n\n"
        "    git clone https://github.com/p-e-w/heretic lab/vendor/heretic\n"
    )
    sys.exit(1)

from exercises.lesson_15 import ANSWERS  # noqa: E402

check(isinstance(ANSWERS, dict) and len(ANSWERS) >= 4, "ANSWERS is a dict with at least 4 entries")

for question, citation in ANSWERS.items():
    check(
        isinstance(citation, str) and "???" not in citation,
        f"{question}: still has a placeholder answer — fill it in",
    )
    check(
        "::" in citation,
        f"{question}: citation {citation!r} is in 'file.py::name' format",
    )

    file_part, name_part = citation.split("::", 1)
    cited_file = VENDOR_DIR / file_part

    check(
        cited_file.is_file(),
        f"{question}: cited file {file_part!r} exists under lab/vendor/heretic/src/heretic/",
    )

    content = cited_file.read_text(encoding="utf-8", errors="ignore")
    check(
        name_part in content,
        f"{question}: {name_part!r} appears in {file_part}",
    )

print("Lesson 15: all checks passed.")
