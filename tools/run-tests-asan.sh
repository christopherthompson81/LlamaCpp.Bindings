#!/usr/bin/env bash
# Run the test suite with AddressSanitizer + UndefinedBehaviorSanitizer.
#
# PREREQUISITE: ASan-built llama.cpp libs must be staged into runtimes/.
#   python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_asan/bin --platform linux-x64
#
# To restore the CUDA build for normal dev:
#   python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_cuda/bin --platform linux-x64
#
# Usage:
#   tools/run-tests-asan.sh                   # full suite
#   tools/run-tests-asan.sh "FullyQualifiedName~Backend_Initializes"   # filtered
#
# ASan reports go to stderr AND to /tmp/asan.<pid>.log (one file per process).
# Why: a single sanitizer report can be lost in dotnet test's noisy output;
# the file copy survives even if the process exits abnormally.

set -u  # not -e: we want to keep going past a non-zero exit from dotnet test
        # so the user sees the asan logs that explain WHY it failed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LIBASAN="$(gcc -print-file-name=libasan.so 2>/dev/null)"
if [[ ! -f "$LIBASAN" ]]; then
    # fall back to the SONAME the lib was actually linked against
    LIBASAN="/lib/x86_64-linux-gnu/libasan.so.8"
fi
if [[ ! -f "$LIBASAN" ]]; then
    echo "error: libasan not found (tried gcc and /lib/x86_64-linux-gnu/libasan.so.8)" >&2
    exit 1
fi

# Sanity check: is the staged libllama.so actually the ASan build?
# We detect it by checking whether libllama.so links libasan — the CUDA
# build does not.
STAGED_LLAMA="$REPO_ROOT/runtimes/linux-x64/native/libllama.so"
if [[ ! -f "$STAGED_LLAMA" ]]; then
    echo "error: $STAGED_LLAMA missing — stage native libs first" >&2
    exit 1
fi
if ! readelf -d "$(readlink -f "$STAGED_LLAMA")" 2>/dev/null | grep -q libasan; then
    echo "error: staged libllama.so does not link libasan." >&2
    echo "       You probably have the CUDA build staged. Run:" >&2
    echo "         python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_asan/bin --platform linux-x64" >&2
    exit 1
fi

rm -f /tmp/asan.*.log

export LD_PRELOAD="$LIBASAN"

# detect_leaks=0:        CLR leaks on shutdown by design. Out of scope here.
# halt_on_error=0:       keep going past the first finding so we see them all.
# abort_on_error=0:      let the .NET process exit cleanly (or as cleanly as it can).
# log_path=/tmp/asan:    survives even if dotnet test eats stderr.
# print_stats=1:         summary at the end.
# strict_string_checks=1: catch sloppy strncpy/strncat in our marshalling.
# detect_stack_use_after_return=1: catches the obvious frame-pointer bug class.
export ASAN_OPTIONS="detect_leaks=0:halt_on_error=0:abort_on_error=0:log_path=/tmp/asan:print_stats=1:strict_string_checks=1:detect_stack_use_after_return=1"

# print_stacktrace=1: UBSan reports without stacks are useless.
# halt_on_error=0:    consistent with ASan.
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"

# Disable .NET's minidumps — they fight ASan over the SEGV handler.
export DOTNET_DbgEnableMiniDump=0

echo "ASan harness:"
echo "  LD_PRELOAD = $LD_PRELOAD"
echo "  ASAN_OPTIONS = $ASAN_OPTIONS"
echo "  UBSAN_OPTIONS = $UBSAN_OPTIONS"
echo "  staged libllama.so -> $(readlink -f "$STAGED_LLAMA")"
echo

if [[ $# -ge 1 ]]; then
    dotnet test --nologo --filter "$1"
else
    dotnet test --nologo
fi
EXIT=$?

echo
echo "ASan log files in /tmp:"
ls -la /tmp/asan.*.log 2>/dev/null || echo "  (none — no findings, or process killed before write)"

exit $EXIT
