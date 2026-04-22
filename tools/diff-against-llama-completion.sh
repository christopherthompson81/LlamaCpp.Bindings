#!/usr/bin/env bash
# Differential test driver: run the same prompt + sampler config through
# llama-completion (the reference) and through our binding's diff-runner,
# diff the generated text byte-for-byte, exit nonzero on divergence.
#
# Why: catches logic-level divergence in our binding's sample/decode/accept
# sequencing that wouldn't surface as a memory bug or test-assertion bug —
# i.e., the class that the Phase-3 double-accept lived in until grammar
# (a strict consumer) finally exposed it.
#
# Usage:
#   tools/diff-against-llama-completion.sh \
#       --prompt "..." [--seed 42] [--max-tokens 20] \
#       [--temp 0] [--top-k 1] [--top-p 0.95] [--min-p 0.05] \
#       [--llama-completion PATH] [--model PATH]
#
# Defaults match a deterministic CPU run on TinyLlama (no model downloads
# needed beyond what the test fixtures already pull). Per
# memory/cuda_moe_determinism.md, CUDA + MoE breaks bit-reproducibility;
# CPU + greedy + small model is reproducible. Stay in that regime.
#
# Exit codes:
#   0  outputs match
#   1  outputs diverge (test failed — bug surfaced)
#   2  setup error (missing binary, missing model, invocation problem)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROMPT="The capital of France is"
SEED=42
MAX_TOKENS=20
TEMP="0"
TOP_K="1"
TOP_P="0.95"
MIN_P="0.05"
REPEAT_PENALTY="1.0"
FREQ_PENALTY="0.0"
PRES_PENALTY="0.0"
REPEAT_LAST_N="64"
DRY_MULTIPLIER="0.0"
DRY_BASE="1.75"
DRY_ALLOWED="2"
DRY_LAST_N="-1"
MODEL="/mnt/data/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_COMPLETION="$HOME/Programming/llama.cpp/build_cuda/bin/llama-completion"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)            PROMPT="$2"; shift 2;;
        --seed)              SEED="$2"; shift 2;;
        --max-tokens)        MAX_TOKENS="$2"; shift 2;;
        --temp)              TEMP="$2"; shift 2;;
        --top-k)             TOP_K="$2"; shift 2;;
        --top-p)             TOP_P="$2"; shift 2;;
        --min-p)             MIN_P="$2"; shift 2;;
        --repeat-penalty)    REPEAT_PENALTY="$2"; shift 2;;
        --frequency-penalty) FREQ_PENALTY="$2"; shift 2;;
        --presence-penalty)  PRES_PENALTY="$2"; shift 2;;
        --repeat-last-n)     REPEAT_LAST_N="$2"; shift 2;;
        --dry-multiplier)    DRY_MULTIPLIER="$2"; shift 2;;
        --dry-base)          DRY_BASE="$2"; shift 2;;
        --dry-allowed-length) DRY_ALLOWED="$2"; shift 2;;
        --dry-penalty-last-n) DRY_LAST_N="$2"; shift 2;;
        --model)             MODEL="$2"; shift 2;;
        --llama-completion)  LLAMA_COMPLETION="$2"; shift 2;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | head -40
            exit 0;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

[[ -x "$LLAMA_COMPLETION" ]] || { echo "error: $LLAMA_COMPLETION not executable" >&2; exit 2; }
[[ -f "$MODEL" ]]            || { echo "error: model not found: $MODEL" >&2; exit 2; }

# Build the runner once if needed; subsequent calls reuse.
dotnet build tools/diff-runner/diff-runner.csproj --nologo --verbosity quiet >/dev/null 2>&1 || {
    echo "error: diff-runner build failed; rerun with verbose dotnet" >&2; exit 2;
}

REF_OUT="$(mktemp)"
REF_ERR="$(mktemp)"
BIND_OUT="$(mktemp)"
BIND_ERR="$(mktemp)"
trap 'rm -f "$REF_OUT" "$REF_ERR" "$BIND_OUT" "$BIND_ERR"' EXIT

echo "config:"
echo "  prompt      = $PROMPT"
echo "  seed        = $SEED   (irrelevant when temp=0)"
echo "  max-tokens  = $MAX_TOKENS"
echo "  temp        = $TEMP"
echo "  top-k       = $TOP_K"
echo "  top-p       = $TOP_P"
echo "  min-p       = $MIN_P"
echo "  model       = $MODEL"
echo

echo "1) llama-completion ..."
"$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$MAX_TOKENS" -s "$SEED" \
    --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P" \
    --repeat-penalty "$REPEAT_PENALTY" --frequency-penalty "$FREQ_PENALTY" \
    --presence-penalty "$PRES_PENALTY" --repeat-last-n "$REPEAT_LAST_N" \
    --dry-multiplier "$DRY_MULTIPLIER" --dry-base "$DRY_BASE" \
    --dry-allowed-length "$DRY_ALLOWED" --dry-penalty-last-n "$DRY_LAST_N" \
    -no-cnv -st --no-display-prompt -ngl 0 --no-warmup --no-perf \
    >"$REF_OUT" 2>"$REF_ERR"
REF_EXIT=$?
if (( REF_EXIT != 0 )); then
    echo "  llama-completion exited $REF_EXIT" >&2
    tail -20 "$REF_ERR" >&2
    exit 2
fi

# With --no-display-prompt, llama-completion writes only the generated text to
# stdout. The binding's GEN_TEXT line is also "just the generation". One
# difference: when the model emits EOS, llama-completion writes a literal
# " [end of text]" sentinel to stdout; the binding stops cleanly without one.
# Strip the sentinel for an apples-to-apples content comparison.
REF_GEN="$(cat "$REF_OUT")"
REF_GEN="${REF_GEN% \[end of text\]}"

echo "2) diff-runner ..."
dotnet run --project tools/diff-runner --no-build -- \
    --model "$MODEL" --prompt "$PROMPT" \
    --seed "$SEED" --max-tokens "$MAX_TOKENS" \
    --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P" \
    --repeat-penalty "$REPEAT_PENALTY" --frequency-penalty "$FREQ_PENALTY" \
    --presence-penalty "$PRES_PENALTY" --repeat-last-n "$REPEAT_LAST_N" \
    --dry-multiplier "$DRY_MULTIPLIER" --dry-base "$DRY_BASE" \
    --dry-allowed-length "$DRY_ALLOWED" --dry-penalty-last-n "$DRY_LAST_N" \
    --add-special --gpu-layers 0 \
    >"$BIND_OUT" 2>"$BIND_ERR"
BIND_EXIT=$?
if (( BIND_EXIT != 0 )); then
    echo "  diff-runner exited $BIND_EXIT" >&2
    tail -20 "$BIND_ERR" >&2
    exit 2
fi

# diff-runner emits GEN_TEXT="..." JSON-escaped. Decode the value.
BIND_GEN_RAW="$(awk -F'GEN_TEXT=' '/^GEN_TEXT=/ { print $2; exit }' "$BIND_OUT")"
# Decode JSON string. python is the cleanest portable way.
BIND_GEN="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()), end="")' <<<"$BIND_GEN_RAW")"

# byte-for-byte comparison
if [[ "$REF_GEN" == "$BIND_GEN" ]]; then
    echo
    echo "MATCH (${#REF_GEN} bytes)"
    echo "  generated: $(printf '%q' "$REF_GEN" | head -c 200)"
    exit 0
fi

echo
echo "DIVERGENCE"
echo "  ref  (${#REF_GEN} bytes): $(printf '%q' "$REF_GEN")"
echo "  bind (${#BIND_GEN} bytes): $(printf '%q' "$BIND_GEN")"
echo
echo "first-diff offset: "
python3 - <<EOF
ref = $(printf '%s' "$REF_GEN" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
bnd = $(printf '%s' "$BIND_GEN" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
n = min(len(ref), len(bnd))
i = next((k for k in range(n) if ref[k] != bnd[k]), n)
print(f"  pos {i}: ref={ref[i:i+20]!r} bind={bnd[i:i+20]!r}")
EOF

exit 1
