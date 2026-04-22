# Samples

Two runnable samples live under `samples/`. They're the fastest way to confirm the binding works end-to-end on your machine and serve as reference code for integrating the binding into your own app.

Both expect a GGUF model at `/mnt/data/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` by default — override with `--model` (CLI) or the text box (GUI).

## `samples/LlamaChat.Cli` — terminal REPL

**When to use this:** diagnosing load failures, sanity-checking native lib resolution, exercising the binding without UI complications.

```bash
dotnet run --project samples/LlamaChat.Cli -- \
    --model /path/to/model.gguf \
    --ctx 2048 \
    --temp 0.7 \
    --seed 42 \
    --verbose
```

**Flags:**
- `--model PATH` — GGUF file (default: Qwen3.6-35B path above)
- `--ctx N` — context window in tokens (default 2048)
- `--temp F`, `--seed N` — sampler settings
- `--max N` — per-turn token budget (default 512)
- `--gpu-layers N` — `-1` for all, `0` for CPU-only (default `-1`)
- `--verbose` — route every native log line to stderr, not just warnings/errors

**In-session commands:**
- `/clear` — drop conversation history + KV cache
- `/help` — list commands
- `/quit` or `/exit` — stop
- `Ctrl+C` during generation — cancel the current turn
- `Ctrl+C` at the prompt — exit

**What to look for in its output:**
- `Loading ...` line, then `Model: layers=N, n_embd=N, training_ctx=N` — model load succeeded.
- `Context: n_ctx=2048 (requested 2048)` — KV cache allocated.
- `[native:Warn]` and `[native:Error]` lines surface from llama.cpp's log callback. These are your first diagnostic signal when something fails.

Reading these lines is how we caught the "recurrent memory" mode for Qwen3's Gated Delta Net attention — and it's how to debug any future load failure on a new model or backend.

### Minimal source

Start with [`samples/LlamaChat.Cli/Program.cs`](../samples/LlamaChat.Cli/Program.cs) — it's a single file, ~200 lines, no framework overhead. The state machine is:

1. Parse flags → `Options` record
2. `LlamaBackend.Initialize(logSink: ...)` — register the log sink early so load-time errors are captured
3. Construct `LlamaModel` + `LlamaContext`
4. Read a line from stdin; if it starts with `/`, dispatch a command; otherwise push to history
5. Re-render the prompt via `LlamaChatTemplate.Apply(...)`, call `ClearKvCache()`, build a sampler, stream via `GenerateAsync`
6. Cancel via a `CancellationTokenSource` linked to `Ctrl+C`

## `samples/LlamaChat` — Avalonia MVVM desktop chat

**When to use this:** seeing the binding drive an actual GUI, cribbing ViewModel patterns for your own app.

```bash
dotnet run --project samples/LlamaChat
```

Opens a 900×650 window with a model-path input, chat history pane, sampling controls (temperature, seed, max tokens), input box, and Send / Cancel / Clear / Load-Unload buttons.

**What it demonstrates:**
- A bindable `MainWindowViewModel` that owns `LlamaModel`, `LlamaContext`, and per-turn `LlamaSampler` instances
- `IAsyncEnumerable<string>` streaming into a bound `ChatMessageViewModel.Content` property — the UI updates per-token without manual dispatch
- `CancellationTokenSource` wired to a Cancel button
- Native log routing via `Dispatcher.UIThread.Post` — the log callback fires from llama.cpp threads, and we marshal Warn/Error messages to the status line
- CommunityToolkit.Mvvm source generators (`[ObservableProperty]`, `[RelayCommand]`) with the **non-obvious** pairing of `[NotifyPropertyChangedFor]` + `[NotifyCanExecuteChangedFor]` for source-of-truth fields that feed a command's `CanExecute` predicate

**What it doesn't do (yet):**
- Persist conversations
- Load-on-startup (you click Load each time)
- Stream-processing of `<think>...</think>` blocks — they render as literal tags in Qwen3 output. A real chat UI would either strip them for display or collapse them into an expandable "thinking" affordance.
- VRAM-aware context/offload heuristics — the context size is hardcoded to 2048 and GPU layers to `-1`. See the "hardware heuristics" thread in [kickoff.md](kickoff.md) and [version_pin memory](../../.claude/projects/-home-chris-Programming-LlamaCpp-Bindings/memory/version_pin.md) for the planned approach.

### When it's also useful: MVVM gotchas

Two patterns worth absorbing if you're writing your own Avalonia host:

1. **Background-thread `[ObservableProperty]` assignment can break things.** The load path does native work on `Task.Run` but assigns the resulting handles to `LoadedModel` / `LoadedContext` back on the UI thread, after the Task completes. Writing to an `[ObservableProperty]` from a background thread fires `PropertyChanged` / `CanExecuteChanged` on that thread, which some downstream Avalonia paths don't handle cleanly.

2. **Button `IsEnabled` needs both notification channels.** CTK.Mvvm has two independent notification systems: `INotifyPropertyChanged` (via `[NotifyPropertyChangedFor]`) and `ICommand.CanExecuteChanged` (via `[NotifyCanExecuteChangedFor]`). A field that feeds a `[RelayCommand(CanExecute = ...)]` predicate needs **both** attributes; with only the first, the button binds `IsEnabled` once at construction time (when `CanExecute` was false) and never refreshes.

## Running both samples in sequence

If you run the Avalonia app and then try `dotnet test` without closing the window, the xUnit fixture will fail to allocate a second copy of the 17 GB model in VRAM. Close the GUI first — `/quit` in the CLI, or close the window button in the GUI. Nothing special about our code; just GPU budget arithmetic.
