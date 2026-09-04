# SPDX-License-Identifier: MIT
"""Imported by every checks/NN.py. Puts lab/ on sys.path (so `lib` and
`exercises` import cleanly regardless of cwd) and gives every checker the
same PASS/FAIL vocabulary.
"""

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def check(condition: bool, msg: str) -> None:
    ok(msg) if condition else fail(msg)
