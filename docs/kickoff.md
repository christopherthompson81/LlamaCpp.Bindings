# Custom llama.cpp C# Binding — Project Kickoff

> **Status:** This document captured the project's original design intent and phase plan. Phases 0–6 were executed as written and are all complete. It's preserved here as a design-rationale reference — useful for understanding *why* the binding looks the way it does — not as a to-do list.
>
> For day-to-day use, start with the [README](../README.md) or [docs/getting-started.md](getting-started.md). For the maintenance workflow when llama.cpp updates, see [docs/updating-llama-cpp.md](updating-llama-cpp.md).

## Project goal

Build a minimal, maintainable C# binding layer for `llama.cpp` that can be used from an Avalonia desktop application for in-process LLM inference. The binding should be small enough to understand end-to-end and updateable in an afternoon when `llama.cpp` changes.

**Not goals:** Full coverage of the `llama.cpp` C API, multi-user batching, training, fine-tuning, embedding servers, or anything that `llama.cpp`'s own `llama-server` already does well.

**In scope:** Loading one GGUF model, running a streaming generation loop with modern sampler chains, surfacing tokens to the UI via `IAsyncEnumerable<string>`, and handling KV cache lifecycle for a chat-style session.

## Background context

`llama.cpp` publishes a tagged GitHub release for essentially every merged PR (multiple per day), each with prebuilt binaries for Windows/Linux/macOS across CPU, CUDA, Vulkan, Metal, ROCm, and SYCL backends. There is no meaningful stability difference between "master" and "latest release." The project's C API (`llama.h`) changes on a much slower cadence than the release tags — most releases don't touch the public header at all.

This means:
- We don't build native binaries ourselves. We download them from GitHub releases.
- We don't need to pin to a "stable" version. We pin to a specific release tag, and bump it when we want.
- Our maintenance burden is driven by `llama.h` diffs, not by release frequency.

Existing C# bindings (LLamaSharp) are comprehensive but lag llama.cpp by weeks-to-months and carry API surface we don't need. A custom thin binding is a smaller maintenance target for our use case.

## Architecture

Three layers, top to bottom:

1. **Public API** — clean idiomatic C# that Avalonia ViewModels consume. `LlamaModel`, `LlamaContext`, `LlamaGenerator` with streaming methods.
2. **Safe handles** — `SafeHandle` subclasses wrapping every opaque native pointer, ensuring deterministic cleanup and preventing use-after-free.
3. **P/Invoke** — `[LibraryImport]` declarations and `[StructLayout(LayoutKind.Sequential)]` struct mirrors. Internal. Never exposed to consumers.

Alongside this, a **maintenance tooling** layer lives in a separate folder:

4. **Header-diff pipeline** — scripts that fetch `llama.h` from a target llama.cpp release, extract the API surface via libclang, diff it against the pinned version, and emit a change report for Claude Code to act on.

## Repository layout

```
/
├── src/
│   ├── MyApp.Llama/                   # The binding library
│   │   ├── Native/
│   │   │   ├── NativeMethods.cs       # All P/Invoke declarations
│   │   │   ├── NativeStructs.cs       # Struct mirrors for llama.h types
│   │   │   ├── NativeEnums.cs         # Enum mirrors
│   │   │   └── SafeHandles/
│   │   │       ├── SafeLlamaModelHandle.cs
│   │   │       ├── SafeLlamaContextHandle.cs
│   │   │       └── SafeLlamaSamplerHandle.cs
│   │   ├── LlamaModel.cs              # Public API
│   │   ├── LlamaContext.cs
│   │   ├── LlamaGenerator.cs
│   │   ├── LlamaBackend.cs            # Static init/teardown
│   │   └── MyApp.Llama.csproj
│   └── MyApp.Llama.Tests/
│       ├── StructLayoutTests.cs       # Size/offset assertions
│       ├── SmokeTests.cs              # Load model, generate tokens
│       └── MyApp.Llama.Tests.csproj
├── runtimes/                          # Native binaries (gitignored, fetched)
│   ├── win-x64/native/llama.dll
│   ├── linux-x64/native/libllama.so
│   └── osx-arm64/native/libllama.dylib
├── third_party/
│   └── llama.cpp/
│       ├── include/llama.h.pinned     # Committed copy of pinned header
│       └── VERSION                    # e.g. "b8642"
├── tools/
│   ├── fetch-binaries.py              # Downloads release artifacts
│   ├── extract-api.py                 # libclang → JSON API description
│   ├── diff-api.py                    # Two JSONs → change report
│   ├── xref-bindings.py               # Change report × NativeMethods.cs
│   └── check-for-updates.sh           # Orchestrates the above
├── CLAUDE.md                          # Project conventions for Claude Code
└── LLAMA_BINDING_PROJECT.md           # This file
```

## The ~35-function API surface we need to bind

Grouped by purpose. Function names reflect recent `llama.cpp` naming (post vocab-split, post-sampler-refactor). Verify against the pinned `llama.h` before writing P/Invokes — names may have shifted.

### Backend lifecycle
- `llama_backend_init`
- `llama_backend_free`
- `llama_log_set` (for routing native logs into our logger)
- `llama_numa_init` (optional; NUMA tuning)

### Model loading
- `llama_model_default_params` → returns `llama_model_params` by value
- `llama_model_load_from_file` → returns `llama_model*`
- `llama_model_free`
- `llama_model_get_vocab` → returns `llama_vocab*` (vocab was split out from model)
- `llama_model_n_ctx_train`
- `llama_model_n_embd`
- `llama_model_chat_template` (for auto-detecting chat format from GGUF metadata)

### Context creation
- `llama_context_default_params` → returns `llama_context_params` by value
- `llama_init_from_model` (formerly `llama_new_context_with_model`)
- `llama_free` (frees context)
- `llama_n_ctx`
- `llama_n_batch`

### Vocab and tokenization
- `llama_tokenize`
- `llama_token_to_piece`
- `llama_detokenize` (optional, can be built from `token_to_piece`)
- `llama_vocab_bos`
- `llama_vocab_eos`
- `llama_vocab_is_eog` (is end-of-generation — handles multiple stop tokens)
- `llama_vocab_n_tokens`

### Batch and decode
- `llama_batch_init`
- `llama_batch_free`
- `llama_batch_get_one` (convenience for single-sequence simple cases)
- `llama_decode` — the hot path
- `llama_get_logits_ith`
- `llama_get_logits`

### KV cache
- `llama_kv_self_clear` (formerly `llama_kv_cache_clear`)
- `llama_kv_self_seq_rm`
- `llama_kv_self_seq_cp` (optional; useful for branching/speculative)
- `llama_kv_self_used_cells` (diagnostics)

### Sampling (modern chain-based API)
- `llama_sampler_chain_default_params`
- `llama_sampler_chain_init`
- `llama_sampler_chain_add`
- `llama_sampler_init_top_k`
- `llama_sampler_init_top_p`
- `llama_sampler_init_min_p`
- `llama_sampler_init_temp`
- `llama_sampler_init_dist` (terminal sampler — actually picks a token)
- `llama_sampler_init_penalties` (repetition/frequency/presence)
- `llama_sampler_init_grammar` (optional; constrained generation)
- `llama_sampler_sample`
- `llama_sampler_accept`
- `llama_sampler_reset`
- `llama_sampler_free`

### Chat templating
- `llama_chat_apply_template` (uses the model's embedded Jinja template; much simpler than implementing per-model formatting ourselves)

## Structs we need to mirror

Each of these must match the C header byte-for-byte. Field order, field types, and any padding must be correct or we get silent memory corruption.

- `llama_model_params` — n_gpu_layers, main_gpu, tensor_split pointer, progress callback, kv_overrides pointer, vocab_only, use_mmap, use_mlock, check_tensors
- `llama_context_params` — n_ctx, n_batch, n_ubatch, n_seq_max, n_threads, n_threads_batch, rope_scaling_type, pooling_type, attention_type, rope_freq_base, rope_freq_scale, yarn_* fields, defrag_thold, cb_eval, cb_eval_user_data, type_k, type_v, abort_callback, abort_callback_data, and several bools (embeddings, offload_kqv, flash_attn, no_perf, op_offload, swa_full)
- `llama_batch` — n_tokens, token pointer, embd pointer, pos pointer, n_seq_id pointer, seq_id pointer-of-pointers, logits pointer, all_pos_0, all_pos_1, all_seq_id
- `llama_sampler_chain_params` — no_perf bool
- `llama_token_data` — id, logit, p
- `llama_token_data_array` — data pointer, size, selected, sorted
- `llama_chat_message` — role pointer, content pointer

Enums to mirror:
- `llama_vocab_type`
- `llama_rope_type`
- `llama_token_type` / `llama_token_attr`
- `llama_ftype`
- `llama_rope_scaling_type`
- `llama_pooling_type`
- `llama_attention_type`
- `llama_split_mode`

## Non-negotiable conventions

These go in `CLAUDE.md` at the repo root so Claude Code respects them on every edit.

**P/Invoke declarations:**
- Use `[LibraryImport("llama", StringMarshalling = StringMarshalling.Utf8)]`, never `[DllImport]`. This requires `partial` methods and .NET 8+.
- All P/Invokes live in `NativeMethods.cs` as `internal static partial`.
- Function names in C# mirror the C name exactly (e.g. `llama_decode`, not `LlamaDecode`). This makes header diffs trivial to apply.
- For functions that return strings (char*), use `[return: MarshalUsing(typeof(Utf8StringMarshaller))]` and handle null explicitly.

**Structs:**
- `[StructLayout(LayoutKind.Sequential)]` on every mirror struct. No `Auto` or `Explicit` unless absolutely required.
- Field order must match `llama.h` exactly. Do not reorder for "readability."
- Use `nint`/`nuint` for native pointer-sized integers. Use `IntPtr` only for opaque pointers being passed through.
- For C `bool` (which is `_Bool`, 1 byte in llama.cpp), use `[MarshalAs(UnmanagedType.I1)] bool`. Never assume a raw `bool` marshals correctly.
- For fixed-size inline arrays in C structs, use `fixed` buffers inside `unsafe` structs.
- Every struct gets a static `SizeAssertion()` method called at module init that verifies `Marshal.SizeOf<T>()` matches a known-good value. If the assertion fires, we refuse to load rather than silently corrupt memory.

**SafeHandles:**
- Every opaque native pointer (model, context, sampler, batch-if-heap-allocated) has a dedicated `SafeHandle` subclass.
- `ReleaseHandle()` calls the corresponding `_free` function and returns true.
- Public API never exposes raw `IntPtr`. It takes and returns `SafeXxxHandle`.

**Public API:**
- Classes are `IDisposable`. `Dispose()` disposes the underlying `SafeHandle`.
- No static mutable state except the backend init flag.
- Streaming generation is exposed as `IAsyncEnumerable<string>`, yielding decoded piece strings (not raw tokens).
- All paths that can take meaningful time accept a `CancellationToken`.

**Error handling:**
- Native functions that return int status codes get checked; nonzero throws `LlamaException` with the code and function name.
- Native functions that return pointers get null-checked; null throws `LlamaException`.

## Phase plan

Work through these phases in order. Each phase should end in a working, tested state before starting the next.

### Phase 0 — Project skeleton (1 session)
- Create the solution and project structure above
- Add `MyApp.Llama.csproj` targeting `net8.0`, `AllowUnsafeBlocks=true`, `LangVersion=latest`
- Add `MyApp.Llama.Tests.csproj` with xUnit
- Write `CLAUDE.md` capturing the conventions above
- Write `tools/fetch-binaries.py` that takes a release tag and a platform/backend combination and downloads the right archive into `runtimes/<rid>/native/`
- Commit a pinned `llama.h` into `third_party/llama.cpp/include/` and record the tag in `VERSION`

### Phase 1 — Backend, model, context (1 session)
- Implement `LlamaBackend` static class with `Init()` / `Free()` and a static constructor that wires up the log callback
- Implement `SafeLlamaModelHandle`, `SafeLlamaContextHandle`
- Mirror `llama_model_params` and `llama_context_params` structs; write size assertions
- Implement `LlamaModel` class: constructor takes path + params, loads model, exposes `Vocab`, `ContextSize`, `EmbeddingDimension`
- Implement `LlamaContext` class: constructor takes model + params, exposes `NCtx`
- Smoke test: load a small GGUF (TinyLlama works), create a context, dispose everything, no crashes

### Phase 2 — Tokenization and chat template (1 session)
- Bind `llama_tokenize`, `llama_token_to_piece`, `llama_vocab_*` helpers
- Add `LlamaVocab` class with `Tokenize(string)`, `TokenToPiece(int)`, `Bos`, `Eos`, `IsEndOfGeneration(int)`
- Bind `llama_chat_apply_template` and `llama_model_chat_template`
- Add `LlamaChatTemplate` helper that takes a list of `(role, content)` pairs and produces a prompt string
- Test: tokenize a known string, verify round-trip; apply chat template to a simple conversation, verify output matches llama.cpp CLI

### Phase 3 — Decode and sampling (1-2 sessions)
- Bind `llama_batch_*` functions; mirror `llama_batch` struct carefully (it has several pointer-to-pointer fields)
- Bind `llama_decode`, `llama_get_logits_ith`
- Bind the full sampler chain API; create `SafeLlamaSamplerHandle`
- Implement `LlamaSampler` class with a fluent builder: `.WithTopK(40).WithTopP(0.9f).WithTemperature(0.7f).WithMinP(0.05f).WithDistribution(seed)`
- Implement `LlamaGenerator.GenerateAsync(prompt, sampler, cancellationToken)` returning `IAsyncEnumerable<string>`
- Handle EOG detection to end generation cleanly
- Test: generate 50 tokens from a known prompt with a fixed seed, verify output is coherent

### Phase 4 — KV cache management (1 session)
- Bind `llama_kv_self_clear`, `llama_kv_self_seq_rm`
- Add `Clear()` and `RemoveSequence(int, int, int)` methods on `LlamaContext`
- Verify multi-turn chat works: prompt 1 → response 1 → prompt 2 (with history) → response 2, without re-processing the full history each turn

### Phase 5 — Maintenance tooling (1-2 sessions)
- Write `tools/extract-api.py` using `clang.cindex` to produce a JSON description of `llama.h`: all exported functions (name, return type, parameters) and all structs (name, fields with types)
- Write `tools/diff-api.py` that takes two such JSON files and emits a markdown change report with sections for added/removed/renamed/signature-changed functions and struct layout changes
- Write `tools/xref-bindings.py` that scans `NativeMethods.cs` and `NativeStructs.cs`, and for each change in the report notes which C# files reference the affected symbol
- Write `tools/check-for-updates.sh`: fetches the latest release tag, downloads its `llama.h`, runs the full pipeline, emits `UPDATE_REPORT.md`
- Test the full loop by intentionally diffing against an older tag and reviewing the generated report

### Phase 6 — Avalonia integration example (1 session)
- Add a tiny Avalonia app in `samples/` with a text input, a streaming output pane, and a model picker
- Wire `LlamaGenerator.GenerateAsync` into the ViewModel with proper cancellation on user interrupt
- This doubles as end-to-end validation that the API is actually pleasant to use

## Ongoing update workflow

Once Phase 5 is done, the day-to-day loop looks like:

1. Run `tools/check-for-updates.sh` (can be weekly cron, manual, or CI)
2. If `UPDATE_REPORT.md` is empty: update `VERSION`, refresh binaries, commit, ship
3. If not empty: feed the report + current `NativeMethods.cs` / `NativeStructs.cs` to Claude Code with a prompt like "apply the changes in UPDATE_REPORT.md to the native layer, showing me each struct diff before committing"
4. Review each struct change manually (this is the highest-risk category)
5. Update `llama.h.pinned` to the new version
6. Run smoke tests
7. Commit

Most updates will be no-ops. Real ABI changes happen on a roughly quarterly cadence.

## First task for Claude Code

Start with Phase 0. Specifically:

1. Scaffold the solution and projects as laid out above
2. Write `CLAUDE.md` capturing the non-negotiable conventions
3. Write `tools/fetch-binaries.py` with this interface:
   ```
   python tools/fetch-binaries.py --tag b8642 --platform win-x64 --backend cuda-12.4
   ```
   It should download the matching archive from `https://github.com/ggml-org/llama.cpp/releases/download/<tag>/llama-<tag>-bin-win-cuda-12.4-x64.zip`, extract only the `.dll`/`.so`/`.dylib` files (not the CLI tools), and place them in `runtimes/<rid>/native/`.
4. Run the script once for the developer's current platform, confirm binaries land in the right place
5. Download and commit `llama.h` from the chosen tag into `third_party/llama.cpp/include/llama.h.pinned`, and write the tag into `third_party/llama.cpp/VERSION`

Do not start Phase 1 yet. Stop after Phase 0 so we can verify the foundation is right.

## Open questions — status

**Target platforms (resolved, 2026-04-21):**
- Dev workstation is Linux x64 + NVIDIA 3090 + CUDA 12.8 driver. This is also the primary test rig.
- Upstream ships no Linux CUDA prebuilt (scanned releases b2000..b8875, none). Options were: Vulkan prebuilt (works on 3090, ~80-90% CUDA perf, zero user setup), ROCm (AMD only), or a locally-built CUDA lib.
- **Decision:** use the locally-built CUDA lib from `~/Programming/llama.cpp/build_cuda/bin` for now. `tools/fetch-binaries.py` supports both release mode (`--tag b#### --backend vulkan`) and local-build mode (`--from-local <dir>`).
- Distribution to others is deferred. When we tackle it, the likely shape is: Windows → release CUDA 12.4 prebuilt, macOS → release Metal prebuilt, Linux → either upstream Vulkan prebuilt as default with optional self-hosted CUDA bundle, or a self-hosted CUDA prebuilt mirrored in our own GitHub releases.

**Pinned upstream version (resolved, 2026-04-21):**
- Pinned to the local CUDA build: git `b8610-10-g1d6d4cf7a` (SOVERSION `libllama.so.0.0.8620`, header dated 2026-04-01).
- **Not bumping to the latest release (b8875) because b8875 removed two APIs we want to lean on:** `llama_params_fit()` (auto-fits model + context params to available device memory — matches our stated hardware-heuristics goal) and `llama_memory_breakdown_print()` (per-device memory diagnostics). b8875 adds `LLAMA_FTYPE_MOSTLY_Q1_0` and `LLAMA_SPLIT_MODE_TENSOR`, neither of which we need today.
- **Risk to track:** `llama_params_fit` being removed upstream suggests it was experimental or inadequate. If we lean on it in the binding, we are on an ABI branch. Mitigation: keep a fallback path that implements our own VRAM-budgeting heuristic in pure managed code, so if/when we have to upgrade past the removal, we swap implementations without changing the public API.

**Target model (resolved, 2026-04-21):**
- Golden model for the dev workstation: `/mnt/data/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (17.7 GB, Qwen3 MoE with 3B active params, IQ4_XS quant — picked to fit the 3090's 24GB with headroom for KV cache).
- Note that the kickoff's Phase 1 smoke test calls for "something small like TinyLlama Q4_K_M (<2GB)". For fast CI loops, having a tiny secondary model is still worthwhile — Qwen3.6-35B takes tens of seconds just to mmap. TBD whether we add TinyLlama as a separate CI fixture.
- End-user configuration is intended to be heuristics-driven — pick a GGUF that fits the user's VRAM/RAM budget rather than hard-coding. Leans on `llama_params_fit` (see above).

**Logging (unresolved):**
- What logging framework will the host Avalonia app use? The native `llama_log_set` callback needs a destination. Likely `Microsoft.Extensions.Logging` given it's the .NET default, but confirm once the host app exists.

**.NET target framework (resolved, 2026-04-21):**
- Kickoff asked for `net8.0`. Dev workstation has .NET 10 SDK only and no net8 targeting pack. Targeting `net10.0` — all APIs we depend on (`LibraryImport`, partial P/Invokes, `Utf8StringMarshaller`) were introduced in net7 and work identically. Easy to downgrade when we have deployment targets that require net8.

**Project naming (resolved, 2026-04-21):**
- Using `LlamaCpp.Bindings` (matching the repo folder) instead of the kickoff's placeholder `MyApp.Llama`.