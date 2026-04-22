# LlamaCpp.Bindings — Conventions for Claude Code

This is a thin, hand-rolled C# binding over `llama.cpp`'s C API for in-process LLM inference from an Avalonia desktop app. The design premise and phase plan live in `docs/kickoff.md` at the repo root — read that first before making cross-cutting changes.

## Project identity

- Library: `LlamaCpp.Bindings` (root namespace and assembly). The kickoff used placeholder `MyApp.Llama`; we use the repo name instead.
- Tests: `LlamaCpp.Bindings.Tests` (xUnit).
- Solution: `LlamaCpp.Bindings.slnx` (.NET 10 SDK XML-based solution format).

## Target framework

- `net10.0`. The kickoff requested `net8.0`, but the installed SDK is 10.x and no `net8` targeting pack is available on this dev box. The APIs we depend on (`LibraryImport`, partial P/Invokes, `Utf8StringMarshaller`) were introduced in `.NET 7` and work identically on `net10.0`. Downgrading to `net8.0` is a one-line change in both csproj files if needed for deployment targets.
- `LangVersion=latest`, `Nullable=enable`, `AllowUnsafeBlocks=true`, `TreatWarningsAsErrors=true`.

## Native binaries

- We **do not** build `llama.cpp` ourselves. Binaries are fetched from GitHub release assets by `tools/fetch-binaries.py`.
- Binaries land in `runtimes/<rid>/native/*.{dll,so,dylib}` and are gitignored.
- Pinned version lives in `third_party/llama.cpp/VERSION` (e.g. `b8875`). The matching `llama.h` is committed at `third_party/llama.cpp/include/llama.h.pinned` and is the source of truth for our struct/enum/function mirrors.

## P/Invoke rules (non-negotiable)

- Use `[LibraryImport("llama", StringMarshalling = StringMarshalling.Utf8)]`, never `[DllImport]`. Requires `partial` methods.
- All P/Invokes live in `src/LlamaCpp.Bindings/Native/NativeMethods.cs` as `internal static partial`.
- **C# method names mirror C names exactly** (e.g. `llama_decode`, not `LlamaDecode`). This is what makes header diffs mechanical to apply.
- For `char*` return values, use `[return: MarshalUsing(typeof(Utf8StringMarshaller))]` and null-check explicitly.

## Struct mirror rules

- `[StructLayout(LayoutKind.Sequential)]` on every mirror struct. Never `Auto`. `Explicit` only if the C header forces it.
- Field order must match `llama.h` exactly. **Do not reorder for readability.**
- Native pointer-sized integers use `nint` / `nuint`. Opaque pointers that are passed through unchanged use `IntPtr`.
- C `_Bool` is 1 byte in `llama.cpp` — mirror it as `[MarshalAs(UnmanagedType.I1)] bool`. Never a bare `bool`.
- Fixed-size inline C arrays → `fixed` buffers inside `unsafe` structs.
- Every mirror struct gets a static `SizeAssertion()` method, called at module init, verifying `Marshal.SizeOf<T>()` matches a known value taken from a native binary of the pinned version. If an assertion fails, refuse to load rather than silently corrupt memory.

## SafeHandle rules

- Every opaque native pointer (`llama_model*`, `llama_context*`, `llama_sampler*`, and heap-allocated `llama_batch` if applicable) gets a dedicated `SafeHandle` subclass under `Native/SafeHandles/`.
- `ReleaseHandle()` calls the corresponding `_free` function and returns `true`.
- **Public API never exposes raw `IntPtr`.** It takes and returns `SafeXxxHandle`.

## Public API rules

- Public classes are `IDisposable`; `Dispose()` disposes the underlying SafeHandle(s).
- No static mutable state except the one-time backend init flag in `LlamaBackend`.
- Streaming generation is exposed as `IAsyncEnumerable<string>`, yielding decoded piece strings (not raw token IDs).
- Every path that can take meaningful time accepts a `CancellationToken`.

## Error handling

- Native functions returning status `int`: nonzero → throw `LlamaException(functionName, code)`.
- Native functions returning pointers: null → throw `LlamaException(functionName)`.
- Never swallow native errors silently.

## Folder layout

```
src/LlamaCpp.Bindings/
  Native/
    NativeMethods.cs        ← all P/Invoke decls, internal static partial
    NativeStructs.cs        ← struct mirrors, sequential layout
    NativeEnums.cs          ← enum mirrors
    SafeHandles/            ← one SafeHandle per opaque pointer type
  LlamaBackend.cs           ← static init/teardown + log routing
  LlamaModel.cs             ← public API
  LlamaContext.cs
  LlamaGenerator.cs
src/LlamaCpp.Bindings.Tests/
  StructLayoutTests.cs      ← size/offset assertions against pinned version
  SmokeTests.cs             ← load tiny GGUF, generate tokens
runtimes/<rid>/native/      ← fetched native binaries (gitignored)
third_party/llama.cpp/
  include/llama.h.pinned    ← committed source of truth
  VERSION                   ← tag string, e.g. "b8875"
tools/
  fetch-binaries.py         ← downloads release artifacts
  extract-api.py            ← libclang → JSON (Phase 5)
  diff-api.py               ← JSON+JSON → markdown change report (Phase 5)
  xref-bindings.py          ← change report × NativeMethods.cs (Phase 5)
  check-for-updates.sh      ← orchestrates the above (Phase 5)
```

## Phase discipline

- We follow the phase plan in `docs/kickoff.md`. Each phase must end in a working, tested state before the next starts. Do not scatter half-finished bindings across phases.
- Currently at: **end of Phase 0**. Phase 1 has not been started.

## Investigation docs

Per global convention: when running experimental or analytical loops (feasibility spikes, parity investigations, struct-layout debugging), maintain a chronological `docs/<topic>_investigation.md`. Expect these when validating struct sizes against native binaries, diagnosing decode divergence from the CLI, and similar — not for routine edits.
