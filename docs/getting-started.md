# Getting started

This page walks through adding LlamaCpp.Bindings to a C# project and running your first generation. For the minute-to-minute development loop, also see [samples.md](samples.md) — the bundled CLI is usually the fastest way to see the binding working.

## 0. Prerequisites

- **.NET SDK 8.0 or newer.** The repo currently targets `net10.0`; `net8.0` works fine if you change the two `TargetFramework` lines. We use three APIs that all landed in `.NET 7`: `[LibraryImport]`, partial P/Invokes, and `Utf8StringMarshaller`.
- **A GGUF model file.** Any quant supported by your llama.cpp build works. Pick one that fits your VRAM/RAM budget — `Qwen2.5-3B-Instruct-Q4_K_M` (~2 GB) is a good first target; `TinyLlama-1.1B-Chat-v1.0-Q4_K_M` (~650 MB) is smaller still if you want a CPU-only test.
- **A native llama.cpp build for your platform.** See [native-binaries.md](native-binaries.md) for the full story; `tools/fetch-binaries.py` automates the prebuilt case.

## 1. Reference the binding

From source (recommended while the library is pre-1.0):

```bash
git clone https://github.com/<owner>/LlamaCpp.Bindings.git
cd my-app
dotnet add reference ../LlamaCpp.Bindings/src/LlamaCpp.Bindings/LlamaCpp.Bindings.csproj
```

There is no NuGet package yet. If/when one ships it'll be named `LlamaCpp.Bindings`.

## 2. Put a native library where .NET will find it

The binding's P/Invoke resolver probes `runtimes/<rid>/native/` next to your executable. The easiest way to populate that directory is to let `tools/fetch-binaries.py` do it:

```bash
# Linux x64, Vulkan backend (works on NVIDIA, AMD, and Intel GPUs)
python tools/fetch-binaries.py --tag b8875 --platform linux-x64 --backend vulkan

# Windows x64, CUDA 12.4 (forward-compatible with CUDA 12.x drivers)
python tools/fetch-binaries.py --tag b8875 --platform win-x64 --backend cuda-12.4 --with-cudart 12.4

# macOS arm64, Metal
python tools/fetch-binaries.py --tag b8875 --platform osx-arm64 --backend metal
```

This downloads the matching archive from [ggml-org/llama.cpp's releases](https://github.com/ggml-org/llama.cpp/releases), extracts just the shared libraries, and writes them under `runtimes/<rid>/native/`.

If you built llama.cpp yourself (common on Linux where no upstream CUDA prebuilt exists), use the local-build mode:

```bash
python tools/fetch-binaries.py --from-local ~/Programming/llama.cpp/build_cuda/bin --platform linux-x64
```

See [native-binaries.md](native-binaries.md) for the per-platform details, caveats, and RUNPATH gotchas.

## 3. First generation

```csharp
using LlamaCpp.Bindings;

// Call once, process-wide. Idempotent.
LlamaBackend.Initialize();

using var model = new LlamaModel(
    "/path/to/your/model.gguf",
    new LlamaModelParameters
    {
        GpuLayerCount = -1,   // all layers on GPU; set 0 for CPU-only
        UseMmap       = true,
    });

using var context = new LlamaContext(
    model,
    new LlamaContextParameters
    {
        ContextSize = 2048,   // tokens of attention window
    });

using var sampler = new LlamaSamplerBuilder()
    .WithTopK(40)
    .WithTopP(0.9f)
    .WithMinP(0.05f)
    .WithTemperature(0.7f)
    .WithDistribution(seed: 42)
    .Build();

var generator = new LlamaGenerator(context, sampler);

await foreach (var piece in generator.GenerateAsync(
    "Write a one-sentence haiku about autumn.",
    maxTokens: 128))
{
    Console.Write(piece);
}
Console.WriteLine();
```

Run it. If the model file and native library are both in place, you'll see tokens stream to stdout.

## 4. A real chat loop

For anything conversational, use the embedded chat template and multi-turn history:

```csharp
var history = new List<ChatMessage>
{
    new("system", "You are a concise assistant."),
    new("user",   "What's the capital of France?"),
};

var template = model.GetChatTemplate()
    ?? throw new InvalidOperationException("Model has no embedded chat template.");

var prompt = LlamaChatTemplate.Apply(template, history, addAssistantPrefix: true);

// Clear the KV cache per turn — simplest correct strategy. See
// LlamaContext.SequencePositionRange / RemoveSequenceRange for a faster
// delta-decode alternative if long histories become a bottleneck.
context.ClearKvCache();

var reply = new StringBuilder();
await foreach (var piece in generator.GenerateAsync(
    prompt,
    maxTokens: 256,
    addSpecial:   false,
    parseSpecial: true))  // template emits <|im_start|> etc. as text; parse them as single tokens
{
    Console.Write(piece);
    reply.Append(piece);
}

history.Add(new ChatMessage("assistant", reply.ToString()));
// ... and loop for the next turn.
```

## 5. Cancellation

Every async path takes a `CancellationToken`. A typical desktop pattern:

```csharp
using var cts = new CancellationTokenSource();
// Wire a Cancel button: cts.Cancel();

try
{
    await foreach (var piece in generator.GenerateAsync(
        prompt, maxTokens: 1024, cancellationToken: cts.Token))
    {
        Console.Write(piece);
    }
}
catch (OperationCanceledException)
{
    Console.WriteLine(" [cancelled]");
}
```

The cancellation is cooperative — it fires between tokens, so the worst-case latency is one decode step (~tens of milliseconds on GPU).

## 6. Where to go next

- **[samples.md](samples.md)** — tour of the bundled CLI and Avalonia samples
- **[native-binaries.md](native-binaries.md)** — platform-specific native-lib details and backend choice
- **[updating-llama-cpp.md](updating-llama-cpp.md)** — how to bump the pinned llama.cpp version
- **[troubleshooting.md](troubleshooting.md)** — `DllNotFoundException`, struct-layout assertion failures, VRAM gotchas, CUDA determinism quirks
- **[kickoff.md](kickoff.md)** — why the binding looks the way it does
