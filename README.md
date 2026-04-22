# LlamaCpp.Bindings

A small, hand-rolled C# binding over [`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s C API for in-process LLM inference. Designed to be small enough to understand end-to-end and updateable in an afternoon when llama.cpp changes.

**In scope**
- Loading one GGUF model
- Streaming generation via `IAsyncEnumerable<string>` with modern chain-based samplers
- Chat templating via the GGUF-embedded Jinja template
- KV cache lifecycle for multi-turn chat

**Not in scope**
- Full coverage of the llama.cpp C API
- Multi-user batching or training
- Anything `llama-server` already does well

If you need a comprehensive, long-stable binding, use [LLamaSharp](https://github.com/SciSharp/LLamaSharp). This library trades surface area for a tight maintenance loop — thin enough that bumping the pinned llama.cpp version is usually a ten-minute diff-and-apply job (see [docs/updating-llama-cpp.md](docs/updating-llama-cpp.md)).

## Status

Phases 0–6 of the original [kickoff plan](docs/kickoff.md) are complete. 85 / 85 tests pass, including an end-to-end generation test against a Qwen3 MoE model on a real NVIDIA GPU.

## Quick start

```csharp
using LlamaCpp.Bindings;

LlamaBackend.Initialize();

using var model = new LlamaModel("/path/to/model.gguf", new LlamaModelParameters
{
    GpuLayerCount = -1,   // all layers on GPU (set 0 for CPU-only)
    UseMmap       = true,
});

using var context = new LlamaContext(model, new LlamaContextParameters
{
    ContextSize = 2048,
});

using var sampler = new LlamaSamplerBuilder()
    .WithTopK(40).WithTopP(0.9f).WithMinP(0.05f).WithTemperature(0.7f)
    .WithDistribution(seed: 42)
    .Build();

var generator = new LlamaGenerator(context, sampler);

await foreach (var piece in generator.GenerateAsync(
    "Hello, who are you?", maxTokens: 128))
{
    Console.Write(piece);
}
```

See [docs/getting-started.md](docs/getting-started.md) for a runnable version with native-binary setup.

## Requirements

- **.NET 8+** (project currently targets `net10.0`; `net8.0` works — see [docs/getting-started.md](docs/getting-started.md))
- **A native llama.cpp build** — prebuilt for Windows/macOS, either a Vulkan prebuilt or your own CUDA build for Linux. See [docs/native-binaries.md](docs/native-binaries.md).
- **A GGUF model file.** Any llama.cpp-supported quant works; pick one that fits your VRAM/RAM budget.

## Repository layout

```
src/
  LlamaCpp.Bindings/          The binding library
    Native/                   P/Invoke declarations, struct mirrors, SafeHandles
    LlamaBackend.cs           Process-wide init + log routing
    LlamaModel.cs             GGUF load, vocab, chat template accessor
    LlamaContext.cs           Per-session state + KV cache management
    LlamaVocab.cs             Tokenize / detokenize / special tokens
    LlamaSampler.cs           Fluent sampler chain builder
    LlamaGenerator.cs         IAsyncEnumerable<string> generation loop
    LlamaChatTemplate.cs      Jinja template renderer wrapper
  LlamaCpp.Bindings.Tests/    xUnit tests (struct layout, tokenize, chat, generate, multi-turn)

samples/
  LlamaChat.Cli/              Minimal console REPL — recommended starting point
  LlamaChat/                  Avalonia MVVM desktop chat sample

third_party/
  llama.cpp/
    include/llama.h.pinned    Committed copy of the pinned llama.h
    VERSION                   Which commit we pin to, and why
    api.pinned.json           Baseline for the diff pipeline

tools/
  fetch-binaries.py           Download native binaries from llama.cpp releases
  extract-api.py              Parse a llama.h into structured JSON via libclang
  diff-api.py                 Compare two JSON snapshots -> markdown report
  xref-bindings.py            Cross-reference diff against our C# sources
  check-for-updates.sh        Orchestrator: fetch latest -> diff -> xref -> report
  dump-struct-sizes.{c,sh}    Emit ground-truth struct sizes/offsets

docs/
  kickoff.md                  Original design intent + phase plan (historical)
  getting-started.md          Minimal runnable example
  samples.md                  Tour of the samples
  native-binaries.md          Platform-by-platform native lib story
  updating-llama-cpp.md       Maintenance workflow when llama.cpp changes
  troubleshooting.md          Known quirks + resolution paths
```

## Design, in one paragraph

Three layers, top to bottom: a **public API** of idiomatic `IDisposable` classes (`LlamaModel`, `LlamaContext`, `LlamaSampler`, `LlamaGenerator`); a **SafeHandle layer** that owns every opaque native pointer and guarantees release; a **P/Invoke layer** with `[LibraryImport]` declarations and bit-exact `[StructLayout(Sequential)]` mirrors of every struct we pass to or from native. Every mirrored struct asserts its byte size at module init against values captured by a C size probe — if the native ABI has drifted, we refuse to load rather than silently corrupt memory. Public C# method names mirror the C names exactly (`llama_decode`, not `LlamaDecode`) so header diffs apply mechanically.

Alongside the binding, a **maintenance pipeline** (`tools/`) parses `llama.h` with libclang, diffs it against the pinned snapshot, and cross-references each change against our C# sources — so bumping the pinned version is a guided edit, not archaeology.

## License

MIT. The pinned copy of `third_party/llama.cpp/include/llama.h.pinned` is itself MIT-licensed; see [third_party/llama.cpp/LICENSE](third_party/llama.cpp/LICENSE) for the upstream attribution.
