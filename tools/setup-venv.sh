#!/usr/bin/env bash
# Create the self-contained Python venv the maintenance tooling depends on.
# Idempotent — re-run safe. Gitignored; not shipped.
#
# Tools inside this venv:
#   libclang   — used by extract-api.py to parse llama.h

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet libclang

"$VENV/bin/python" -c "import clang.cindex; print('clang.cindex ok:', clang.cindex.__file__)"
