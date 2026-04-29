#!/usr/bin/env bash
# Build the llamashim native shim against the fetched libllama.so.
# Output lands alongside the existing native binaries in
# runtimes/<rid>/native/, so the bindings load it the same way they
# load libllama.so / libggml-base.so.
#
# Run after fetching binaries (tools/fetch-binaries.py) and re-run on
# every llama.cpp pin bump (the shim depends on libllama internals
# whose ABI may change).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default to linux-x64; override RID for cross-compile or other platforms.
RID="${RID:-linux-x64}"
NATIVE_DIR="${REPO_ROOT}/runtimes/${RID}/native"

if [[ ! -f "${NATIVE_DIR}/libllama.so" ]]; then
    echo "error: ${NATIVE_DIR}/libllama.so not found." >&2
    echo "       Run tools/fetch-binaries.py first." >&2
    exit 1
fi

BUILD_DIR="${SCRIPT_DIR}/build-${RID}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_NATIVE_DIR="${NATIVE_DIR}"
cmake --build "${BUILD_DIR}" --config Release --parallel

# Copy the built shim into runtimes/<rid>/native/ alongside libllama.so
# so the bindings find it via the same NativeLibraryResolver.
case "${RID}" in
    win-*)   SHIM_NAME="llamashim.dll"     ;;
    osx-*)   SHIM_NAME="libllamashim.dylib" ;;
    *)       SHIM_NAME="libllamashim.so"    ;;
esac

cp -f "${BUILD_DIR}/${SHIM_NAME}" "${NATIVE_DIR}/${SHIM_NAME}"
echo "Built ${NATIVE_DIR}/${SHIM_NAME}"
