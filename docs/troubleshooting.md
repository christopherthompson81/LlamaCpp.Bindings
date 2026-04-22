# Troubleshooting

Known quirks and failure modes, with resolution paths. Runs the gamut from "you haven't staged native binaries yet" to "CUDA kernels are non-deterministic for MoE models — don't expect bit-identical greedy output across runs."

## `DllNotFoundException: Unable to load shared library 'llama'`

**Meaning:** .NET's P/Invoke resolver couldn't find the native library.

**What to check (in order):**

1. **Did you call `LlamaBackend.Initialize()` at startup?** The call registers our [`NativeLibraryResolver`](../src/LlamaCpp.Bindings/Native/NativeLibraryResolver.cs), which teaches .NET where to probe. Without it, only the system default search path is used, which won't find `runtimes/<rid>/native/libllama.*`.
2. **Is the native library actually present?** `ls bin/Debug/net*/runtimes/*/native/libllama.*`. If not, run `tools/fetch-binaries.py` for your platform — see [native-binaries.md](native-binaries.md).
3. **Does the library itself load?**
   - Linux: `ldd bin/Debug/net*/runtimes/linux-x64/native/libllama.so` — any red "not found" line reveals a missing dependency (commonly `libcudart.so.12` for CUDA builds).
   - Windows: check that `cudart64_*.dll` sits next to `llama.dll` for CUDA builds. Use `--with-cudart` when fetching.
   - macOS: `otool -L runtimes/osx-arm64/native/libllama.dylib`.
4. **Is the RID right?** Our resolver derives it from `RuntimeInformation.ProcessArchitecture` + `OSPlatform`. If you're on, say, Linux ARM64 but only fetched for `linux-x64`, the probe path won't exist.

## `InvalidOperationException: Struct layout mismatch: llama_model_params is N bytes ... pinned llama.h expects M`

**Meaning:** The native binary at runtime has a different ABI from the `llama.h` we pinned. `NativeLayout.Verify()` caught it at backend init before any P/Invoke could run. Good — the alternative is silent memory corruption.

**Cause:** You swapped the native library (e.g., refetched a newer tag) without updating the pinned header + struct mirrors.

**Fix:**
- If you want the new ABI: follow [updating-llama-cpp.md](updating-llama-cpp.md) to walk the change report and update struct mirrors.
- If you want the old ABI: refetch the native library for the pinned tag (`tools/fetch-binaries.py --tag <pinned-tag>`).

## Model fails to load; VRAM spikes then drops

**Meaning:** The model weights loaded onto the GPU, then *something* after (usually context creation) failed and triggered cleanup — which released VRAM.

**What to check:**

1. **The native log output.** llama.cpp rarely returns a descriptive error code; it logs the real reason through the `log_callback` you registered in `Initialize`. If you're not capturing the log, you're flying blind. For interactive debugging, pass `logSink: (lvl, msg) => Console.Error.WriteLine($"[{lvl}] {msg}")` and look for warnings or errors near the failure point. The CLI sample's `--verbose` flag does exactly this.
2. **Requested context size vs. available VRAM.** If weights took ~20 GB and you asked for a 4096-token full-SWA KV cache on a 24 GB card, the KV allocation can fail. Drop `ContextSize` to 2048 or reduce `GpuLayerCount` to keep some layers on CPU.
3. **Two instances of the model running.** If the Avalonia sample is still holding ~20 GB and you try to run `dotnet test`, the test fixture can't allocate its own copy. Close one side first.

## Tests fail after the GUI sample has run

Same cause as above: the running GUI holds the model in GPU memory. `dotnet test` can't instantiate the shared `GpuGenerationFixture`. Close the app (`/quit` in the CLI, or the window's close button in the GUI), then re-run `dotnet test`.

## Greedy sampling produces different output on consecutive runs (CUDA + MoE)

**Meaning:** CUDA matmul kernels are non-deterministic by default — reduction order varies with kernel scheduling. MoE models (like Qwen3's A3B architecture) amplify this: tiny logit differences pick different experts, which diverge into completely different continuations after a few tokens.

**What to do about it:** test seeded stochastic sampling (`.WithDistribution(seed)`) instead. The RNG dominates the kernel noise, so seeded runs reproduce bit-for-bit. This is why `GenerationTests.Distribution_Generation_With_Fixed_Seed_Is_Reproducible` passes while we *don't* have a greedy-determinism assertion.

If you genuinely need deterministic greedy output:
- Use a dense (non-MoE) model, or
- Run on CPU (slow but deterministic), or
- Configure CUDA for deterministic algorithms (`CUBLAS_WORKSPACE_CONFIG=:4096:8` env var, and use `cudnn_deterministic`-style flags) at the cost of performance.

## `RemoveSequenceRange` returns `false`

**Meaning:** The KV cache backend refused a partial removal. This is not a binding bug — some backends (compact SWA, quantised KV) only support whole-sequence removal. Even `UseFullSwaCache = true` doesn't guarantee arbitrary tail trims succeed on every model.

**What to do:**
- Whole-sequence removal (`fromPosition = 0, toPosition = -1`) always succeeds, on every backend.
- For partial rollback, fall back to `ClearKvCache()` and re-decode the prefix you want to keep.

## `llama_chat_apply_template` fails on a model's embedded template

**Meaning:** llama.cpp's template applier isn't a full Jinja engine — it supports a curated set of known templates (Llama 2/3, ChatML, Mistral, Gemma, Qwen, etc.). Exotic Jinja in GGUF metadata can fail to render.

**What to do:** fall back to manual prompt construction using special-token markers. `LlamaVocab.Eos`, `Eot`, etc. expose the model's terminator tokens; the [Qwen3 chat format](https://qwen.readthedocs.io/en/latest/) uses `<|im_start|>role\ncontent<|im_end|>` which you can assemble by hand. The CLI sample's `RenderPrompt` has a fallback path that does a plain `role: content` join when no template is found — adapt for your model.

## `System.ObjectDisposedException` from a `LlamaVocab` or `LlamaContext`

**Meaning:** You're using an object after its owning model / context was disposed.

**The lifetime rule:** `LlamaContext` must be disposed strictly before its `LlamaModel`. `LlamaVocab` is owned by `LlamaModel` and becomes invalid when the model is disposed. If you hand a reference to a `LlamaVocab` around, track the owning model's lifetime alongside it.

## NumericUpDown / Button doesn't update (Avalonia)

**Meaning:** CommunityToolkit.Mvvm has two independent notification channels — `INotifyPropertyChanged` and `ICommand.CanExecuteChanged`. A field that feeds a `[RelayCommand(CanExecute = ...)]` predicate needs both `[NotifyPropertyChangedFor(nameof(...))]` and `[NotifyCanExecuteChangedFor(nameof(...Command))]` attributes; with only the first, button `IsEnabled` binds once at construction time and never refreshes.

See [`samples/LlamaChat/ViewModels/MainWindowViewModel.cs`](../samples/LlamaChat/ViewModels/MainWindowViewModel.cs) for the correct pairing on `_loadedModel`, `_isBusy`, `_isGenerating`, `_userInput`.

## `n_ctx_seq (X) < n_ctx_train (Y) -- the full capacity of the model will not be utilized`

**Meaning:** Informational warning. The model was trained with a larger context than you're giving it at runtime. Not an error — you'll just get truncated attention compared to what the weights were trained for. Harmless for most chat workloads; relevant if you were planning on long-document summarisation.

**What to do:** raise `LlamaContextParameters.ContextSize` if your VRAM budget allows it. Note that the KV cache is O(context × layers × heads) and grows quickly.

## Something else

If you hit a failure that isn't covered here:

1. Capture the full native log output (`--verbose` in the CLI sample, or the equivalent log sink in your app).
2. Bisect: does the CLI sample fail the same way? If yes, it's a binding or native issue. If no, it's your host code.
3. Check the [project memory](../../.claude/projects/-home-chris-Programming-LlamaCpp-Bindings/) if you're using Claude Code — several quirks are recorded there with full context.
4. File an issue with the native log, your platform / backend, and the `third_party/llama.cpp/VERSION` content.
