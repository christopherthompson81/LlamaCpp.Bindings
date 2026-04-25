# LlamaCpp.Bindings

A comprehensive C# binding suite for the [llama.cpp](https://github.com/ggml-org/llama.cpp) ecosystem: in-process inference, an OpenAI-compatible HTTP server, a desktop chat reference app, and (in progress) tooling for the full GGUF model-prep lifecycle.

The goal is **complete coverage of the llama.cpp surface that real applications need** — every model-load knob, every sampler stage, every advanced feature (speculative decoding, control vectors, LoRA, multimodal, NUMA, tensor-buft overrides) — kept maintainable by a tight pinning + automated diff pipeline rather than letting the wrapper drift incrementally as upstream changes.

## What's in the box

| Project | Purpose |
|---|---|
| **`LlamaCpp.Bindings`** | The core binding library. P/Invokes, struct mirrors, SafeHandles, and idiomatic `IDisposable` wrappers: `LlamaModel`, `LlamaContext`, `LlamaSampler`, `LlamaGenerator`, `LlamaLoraAdapter`, `LlamaControlVector`, `LlamaSpeculativeGenerator`, `MtmdContext` (multimodal), `LlamaHardware` (device enumeration). |
| **`LlamaCpp.Bindings.Server`** | OpenAI-compatible HTTP server with near-feature-complete `llama-server` parity. Chat / completions / embeddings / rerank / tokenize / detokenize, streaming SSE, tool calling, multimodal, multi-session prefix caching, speculative decoding, LoRA + control vectors at startup, Prometheus metrics, API-key auth, TLS, NUMA / device pinning / tensor-buft overrides. See [`docs/server_parity_checklist.md`](docs/server_parity_checklist.md) for the full surface. |
| **`LlamaCpp.Bindings.LlamaChat`** | Avalonia MVVM desktop reference app. Multi-conversation chat, per-conversation exports (Markdown / HTML / PDF / DOCX / XLSX / JSON / TXT), auto-configure heuristics for sizing context against detected VRAM, sampler-profile database, LoRA + speculative-draft selection, state save/load. |
| **`LlamaCpp.Bindings.GGUFSuite`** *(in development)* | The model-prep side of the loop. HuggingFace → GGUF conversion, quantization, control-vector training, perplexity testing — so users can prep a model and immediately serve or chat with it without leaving the C# stack. |
| **`LlamaCpp.Bindings.Tests`** | 375 xUnit tests covering struct layout, tokenization, generation, multi-turn chat, speculative decoding, multimodal, every server endpoint, sampler chains, LoRA, control vectors, embeddings, rerank, NUMA, device pinning. |

## Status

- **Core binding** — every native API the inference + deployment surface needs is wired. Training and `ggml`-graph manipulation are out of scope by design.
- **Server** — 85 features done, 18 tracked under dedicated GitHub issues, 11 declined as explicit non-goals. Remaining items are scope-bounded follow-ups, not gaps. See the [parity checklist](docs/server_parity_checklist.md).
- **LlamaChat desktop app** — usable for daily multi-conversation work.
- **GGUFSuite** — in design. Slot reserved at `src/LlamaCpp.Bindings.GGUFSuite/`.

The pinned llama.cpp version lives in [`third_party/llama.cpp/VERSION`](third_party/llama.cpp/VERSION). Bumping it is usually a guided diff produced by `tools/check-for-updates.sh`, not archaeology.

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

See [docs/getting-started.md](docs/getting-started.md) for a runnable version with native-binary setup. The server has its own getting-started flow under [`src/LlamaCpp.Bindings.Server/`](src/LlamaCpp.Bindings.Server/).

## Requirements

- **.NET 10 SDK** (the projects target `net10.0`; downgrading to `net8.0` is a one-line edit per project file — every API used is `.NET 7+`).
- **A native llama.cpp build** — prebuilt for Windows/macOS, either a Vulkan prebuilt or your own CUDA build for Linux. Fetched and placed by `tools/fetch-binaries.py`. See [docs/native-binaries.md](docs/native-binaries.md).
- **A GGUF model file.** Any llama.cpp-supported quant works; pick one that fits your VRAM/RAM budget.

## Repository layout

```
src/
  LlamaCpp.Bindings/             Core binding library
    Native/                      P/Invoke decls, struct mirrors, SafeHandles
    Jinja/                       Embedded chat-template renderer
    LlamaBackend.cs              Process-wide init, log routing, NUMA
    LlamaModel.cs                GGUF load, vocab, chat template
    LlamaContext.cs              Per-session state, KV cache, LoRA, cvec
    LlamaVocab.cs                Tokenize / detokenize / special tokens
    LlamaSampler.cs              Fluent sampler chain builder
    LlamaGenerator.cs            IAsyncEnumerable<string> generation loop
    LlamaSpeculativeGenerator.cs Two-model speculative decoding
    LlamaLoraAdapter.cs          LoRA adapter loading
    LlamaControlVector.cs        Control-vector loading + attach
    LlamaHardware.cs             Device enumeration + buffer types
    MtmdContext.cs               Multimodal projector
  LlamaCpp.Bindings.Server/      OpenAI-compatible HTTP server
    Endpoints/                   Chat, completion, embeddings, rerank, tokenize, ...
    Services/                    ModelHost, SessionPool, MmprojHost, DraftHost, ...
    Configuration/               ServerOptions
  LlamaCpp.Bindings.LlamaChat/   Avalonia desktop reference app
  LlamaCpp.Bindings.GGUFSuite/   (planned) HF→GGUF, quantize, cvec training, perplexity
  LlamaCpp.Bindings.Tests/       xUnit suite (375 facts)

third_party/
  llama.cpp/
    include/llama.h.pinned       Source of truth for our struct/enum mirrors
    VERSION                      Which commit we pin to, and why
    api.pinned.json              Baseline for the diff pipeline

tools/
  fetch-binaries.py              Download native binaries from llama.cpp releases
  extract-api.py                 Parse llama.h into structured JSON via libclang
  diff-api.py                    Compare two JSON snapshots → markdown report
  xref-bindings.py               Cross-reference diff against our C# sources
  check-for-updates.sh           Orchestrator: fetch latest → diff → xref → report
  dump-struct-sizes.{c,sh}       Emit ground-truth struct sizes/offsets

docs/
  kickoff.md                     Original design intent (historical)
  getting-started.md             Minimal runnable example
  server_parity_checklist.md     Full llama-server feature coverage matrix
  samples.md                     Tour of the reference apps
  native-binaries.md             Platform-by-platform native lib story
  updating-llama-cpp.md          Maintenance workflow when llama.cpp changes
  troubleshooting.md             Known quirks + resolution paths
```

## How comprehensive coverage stays maintainable

Three things working together:

1. **Native-mirror discipline.** Every struct passed across the boundary has a `[StructLayout(Sequential)]` mirror with a byte-size assertion at module init. If the native ABI drifts from the pinned header, we refuse to load rather than silently corrupt memory. Public C# method names mirror C names exactly (`llama_decode`, not `LlamaDecode`) so header diffs apply mechanically.
2. **Pinned versioning.** A specific llama.cpp commit lives in `third_party/llama.cpp/VERSION`. We don't track upstream `main`; we move deliberately when we decide to.
3. **Automated diff pipeline.** `tools/check-for-updates.sh` parses the latest upstream `llama.h` with libclang, diffs it against our pinned snapshot, and cross-references each change against our C# sources. Bumping the pinned version is a guided edit with a punch list, not detective work.

This is how we afford to cover ground that comprehensive bindings normally accumulate over years: we don't promise long-term wrapper stability, we promise that the wrapper tracks upstream tightly and breaks loudly with a clear remediation path when it can't.

## Comparison to LLamaSharp

[LLamaSharp](https://github.com/SciSharp/LLamaSharp) is the de-facto comprehensive C# binding for llama.cpp and has been around longer. The trade-offs differ:

- **LLamaSharp** prioritises long-term wrapper stability — it abstracts native idioms behind a more managed-feeling API and absorbs upstream churn underneath. Best for projects that want to depend on a stable C# surface and not think about llama.cpp's release cadence.
- **This project** prioritises tracking upstream tightly. Public C# names mirror C names; struct layouts are byte-asserted; the diff pipeline forces every upstream change to surface as a guided edit. The result is broader coverage of less-common features and a server that's at near-parity with `llama-server`, at the cost of intentionally breaking when llama.cpp's ABI shifts (the diff pipeline catches these before runtime).

Pick whichever fits your maintenance posture. The ecosystem benefits from both existing.

## License

MIT. The pinned copy of `third_party/llama.cpp/include/llama.h.pinned` is itself MIT-licensed; see [`third_party/llama.cpp/LICENSE`](third_party/llama.cpp/LICENSE) for the upstream attribution.
