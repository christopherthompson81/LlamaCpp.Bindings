#!/usr/bin/env bash
# End-to-end: fetch the latest llama.cpp release tag, extract its API,
# diff against our pinned snapshot, xref against the C# bindings, emit
# UPDATE_REPORT.md at the repo root.
#
# Usage:
#   tools/check-for-updates.sh            # uses latest GitHub release tag
#   tools/check-for-updates.sh b8800      # diffs against a specific tag
#
# Exit codes:
#   0  no API changes (ship without edits)
#   3  API changes detected (see UPDATE_REPORT.md)
#   1+ anything else — failure, investigate stderr

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$SCRIPT_DIR/.venv"
CACHE="$SCRIPT_DIR/.cache/update"
PINNED_JSON="$REPO_ROOT/third_party/llama.cpp/api.pinned.json"
REPORT="$REPO_ROOT/UPDATE_REPORT.md"

mkdir -p "$CACHE"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Bootstrapping tools venv..." >&2
    "$SCRIPT_DIR/setup-venv.sh"
fi

# --- 1. Figure out which tag to compare against ---
if [[ $# -ge 1 ]]; then
    TARGET_TAG="$1"
else
    echo "Resolving latest llama.cpp release tag..." >&2
    TARGET_TAG=$(
        curl -sSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
        | "$VENV/bin/python" -c "import sys,json; print(json.load(sys.stdin)['tag_name'])"
    )
fi
echo "Target tag: $TARGET_TAG" >&2

PINNED_TAG=$(grep -E '^git_describe\s*=' "$REPO_ROOT/third_party/llama.cpp/VERSION" 2>/dev/null \
             | head -1 | sed 's/.*=\s*//' || echo "unknown")
echo "Pinned:     $PINNED_TAG" >&2

# --- 2. Fetch target llama.h + ggml.h (need both for clang parse) ---
TARGET_INCLUDE="$CACHE/$TARGET_TAG/include"
TARGET_GGML="$CACHE/$TARGET_TAG/ggml/include"
mkdir -p "$TARGET_INCLUDE" "$TARGET_GGML"

RAW_BASE="https://raw.githubusercontent.com/ggml-org/llama.cpp/$TARGET_TAG"
echo "Downloading headers from $RAW_BASE ..." >&2
for path in include/llama.h include/llama-cpp.h; do
    curl -sSLf "$RAW_BASE/$path" -o "$CACHE/$TARGET_TAG/$path" || {
        echo "error: failed to fetch $path from $TARGET_TAG" >&2
        exit 1
    }
done
# ggml.h and its siblings — what llama.h transitively includes
for path in ggml/include/ggml.h ggml/include/ggml-cpu.h ggml/include/ggml-backend.h \
            ggml/include/ggml-alloc.h ggml/include/ggml-opt.h ggml/include/gguf.h; do
    # Some files may not exist in older tags; don't fail hard on misses.
    curl -sSLf "$RAW_BASE/$path" -o "$CACHE/$TARGET_TAG/$path" 2>/dev/null \
        || echo "  (skipped $path — not present in $TARGET_TAG)" >&2
done

# --- 3. Extract API JSON for the target tag ---
TARGET_JSON="$CACHE/$TARGET_TAG/api.json"
echo "Extracting API from target header ..." >&2
"$VENV/bin/python" "$SCRIPT_DIR/extract-api.py" \
    "$TARGET_INCLUDE/llama.h" \
    -I "$TARGET_INCLUDE" \
    -I "$TARGET_GGML" \
    -o "$TARGET_JSON"

# --- 4. Ensure a pinned baseline exists ---
if [[ ! -s "$PINNED_JSON" ]]; then
    echo "Pinned API JSON missing; regenerating from llama.h.pinned ..." >&2
    PINNED_HEADER="$REPO_ROOT/third_party/llama.cpp/include/llama.h.pinned"
    # The pinned header was copied from a full checkout; re-use the same ggml includes.
    LOCAL_LLAMA="${LOCAL_LLAMA_CHECKOUT:-$HOME/Programming/llama.cpp}"
    if [[ ! -d "$LOCAL_LLAMA/ggml/include" ]]; then
        echo "error: need ggml includes to parse pinned header. Set LOCAL_LLAMA_CHECKOUT." >&2
        exit 1
    fi
    "$VENV/bin/python" "$SCRIPT_DIR/extract-api.py" \
        "$PINNED_HEADER" \
        -I "$LOCAL_LLAMA/include" \
        -I "$LOCAL_LLAMA/ggml/include" \
        -o "$PINNED_JSON"
fi

# --- 5. Diff + xref ---
DIFF_SECTION="$CACHE/$TARGET_TAG/diff.md"
XREF_SECTION="$CACHE/$TARGET_TAG/xref.md"
"$VENV/bin/python" "$SCRIPT_DIR/diff-api.py" "$PINNED_JSON" "$TARGET_JSON" -o "$DIFF_SECTION"
"$VENV/bin/python" "$SCRIPT_DIR/xref-bindings.py" "$PINNED_JSON" "$TARGET_JSON" \
    --src "$REPO_ROOT/src" -o "$XREF_SECTION"

{
    echo "# llama.cpp update report"
    echo
    echo "- pinned:  \`$PINNED_TAG\`"
    echo "- target:  \`$TARGET_TAG\`"
    echo "- generated: $(date '+%Y-%m-%d %H:%M %Z')"
    echo
    echo "---"
    echo
    cat "$DIFF_SECTION"
    echo
    echo "---"
    echo
    cat "$XREF_SECTION"
} > "$REPORT"

# --- 6. Decide exit code based on whether diff is "empty" ---
# "No API changes" is the explicit phrase emitted by diff-api.py when nothing drifted.
if grep -q '^\*\*No API changes\.\*\*' "$DIFF_SECTION"; then
    echo "No API changes between $PINNED_TAG and $TARGET_TAG." >&2
    echo "Report: $REPORT" >&2
    exit 0
else
    echo "API changes detected. Review: $REPORT" >&2
    exit 3
fi
