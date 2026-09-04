#!/usr/bin/env bash
# The grader. Every lesson's exercise lives in lab/exercises/lesson-NN.py and
# is checked by lab/checks/NN.py against lab/lib — the same library the
# solutions use, so "passing" means your code produces the same tensors the
# reference implementation does, not that it merely runs.
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR"

pick_python() {
  # On stock Windows 11, `python`/`python3`/`py` on PATH are often the
  # Microsoft Store's execution-alias stubs — they exist, `command -v` finds
  # them, and running one either opens the Store or prints "Python was not
  # found" instead of running anything. So don't just check existence: run
  # each candidate and only accept one that actually executes.
  for candidate in python3 python py "$HOME/AppData/Local/Python/bin/python.exe"; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "pass" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  echo "No working Python interpreter found (tried python3, python, py)." >&2
  echo "If you're on Windows and 'python --version' opens the Microsoft Store" >&2
  echo "or says 'Python was not found', disable the App execution alias:" >&2
  echo "  Settings > Apps > Advanced app settings > App execution aliases" >&2
  echo "and turn off the python.exe / python3.exe entries, or install Python" >&2
  echo "from python.org and make sure it comes first on PATH." >&2
  exit 1
}

PY_BIN="$(pick_python)"
VENV_DIR="$LAB_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$VENV_DIR/Scripts/python.exe"

venv_python() {
  if [ ! -x "$VENV_PY" ]; then
    echo "No venv yet — run 'bash lab/lab.sh up' first." >&2
    exit 1
  fi
  echo "$VENV_PY"
}

cmd_up() {
  echo "Creating venv at lab/.venv ..."
  "$PY_BIN" -m venv "$VENV_DIR"
  local venv_py
  venv_py="$(venv_python)"
  echo "Installing requirements (torch, transformers, accelerate) ..."
  "$venv_py" -m pip install --quiet --upgrade pip
  "$venv_py" -m pip install --quiet -r requirements.txt
  echo "Downloading and caching Qwen/Qwen3-0.6B (once, ~1.2GB) ..."
  "$venv_py" -c "from lib.common import load_model, load_tokenizer; load_tokenizer(); load_model(); print('Model cached under lab/.cache')"
  echo "Ready. Try: bash lab/lab.sh check 01"
}

cmd_check() {
  local nn="$1"
  local venv_py
  venv_py="$(venv_python)"
  "$venv_py" "checks/${nn}.py"
}

cmd_hint() {
  local nn="$1"
  echo "Open lessons/module-*/lesson-*.md (check _sidebar.md for lesson ${nn}'s file) and read the '## Hints' section."
  echo "The exercise stub is at lab/exercises/lesson_${nn}.py — every TODO has a comment explaining what's missing."
}

cmd_solve() {
  local nn="$1"
  cp "solutions/lesson_${nn}.py" "exercises/lesson_${nn}.py"
  echo "Copied the reference solution into exercises/lesson_${nn}.py. Re-run: bash lab/lab.sh check ${nn}"
}

cmd_reset() {
  local nn="$1"
  git checkout -- "exercises/lesson_${nn}.py" 2>/dev/null || \
    echo "Could not git-restore exercises/lesson_${nn}.py — restore it from the repo manually."
}

cmd_status() {
  local venv_py
  venv_py="$(venv_python)"
  for f in checks/*.py; do
    nn="$(basename "$f" .py)"
    if "$venv_py" "$f" >/dev/null 2>&1; then
      echo "  [x] lesson $nn"
    else
      echo "  [ ] lesson $nn"
    fi
  done
}

case "${1:-}" in
  up)     cmd_up ;;
  check)  cmd_check "${2:?usage: lab.sh check NN}" ;;
  hint)   cmd_hint "${2:?usage: lab.sh hint NN}" ;;
  solve)  cmd_solve "${2:?usage: lab.sh solve NN}" ;;
  reset)  cmd_reset "${2:?usage: lab.sh reset NN}" ;;
  status) cmd_status ;;
  *)
    echo "Usage: bash lab/lab.sh {up|check NN|hint NN|solve NN|reset NN|status}"
    exit 1
    ;;
esac
