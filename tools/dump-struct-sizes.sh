#!/usr/bin/env bash
# Compile tools/dump-struct-sizes.c against a given llama.cpp include dir
# and emit its JSON output. The .c file includes <llama.h> which transitively
# pulls in ggml.h & friends, so we need a full llama.cpp checkout's include
# tree — the pinned llama.h.pinned alone is not enough.
#
# Usage:
#   tools/dump-struct-sizes.sh <llama.cpp checkout dir>   # writes to stdout
#   tools/dump-struct-sizes.sh                            # defaults to ~/Programming/llama.cpp
#
# The intended workflow is to re-run this when we bump the pinned version
# and update tools/struct-sizes.json in version control, so the
# StructLayoutTests stay in sync with the native binary.

set -euo pipefail

LLAMA_DIR="${1:-$HOME/Programming/llama.cpp}"
if [[ ! -d "$LLAMA_DIR/include" ]] || [[ ! -d "$LLAMA_DIR/ggml/include" ]]; then
    echo "error: not a llama.cpp checkout: $LLAMA_DIR" >&2
    echo "       expected include/ and ggml/include/ subdirs" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/dump-struct-sizes.c"
TMPBIN="$(mktemp)"
trap 'rm -f "$TMPBIN"' EXIT

cc -std=c11 -O0 -Wno-error \
    -I"$LLAMA_DIR/include" \
    -I"$LLAMA_DIR/ggml/include" \
    "$SRC" -o "$TMPBIN"

"$TMPBIN"
