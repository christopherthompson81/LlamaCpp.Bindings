using System.Text;
using LlamaCpp.Bindings;

// Minimal interactive REPL over LlamaCpp.Bindings. Intentionally boring:
// one file, no MVVM, native logs go straight to stderr so we can see what
// llama.cpp is complaining about when things break.
//
// Two modes depending on flags:
//   - Text REPL (default): --model PATH [--ctx N] [--temp F] [--seed N]
//                          [--gpu-layers N] [--verbose]
//   - Multimodal single-shot:
//       --model PATH --mmproj PATH --image PATH [--prompt TEXT]
//       [--image-max-tokens N] [--cpu-encode] [--ctx N] [--gpu-layers N]
//     Exercises the exact MtmdContext/EvalPromptAsync path the desktop app
//     uses, so when something breaks in multimodal we can see the native
//     log on stderr instead of relying on a managed-exception handler.
//
// In-session commands (REPL mode only):
//   /clear    — drop conversation history + KV cache
//   /quit     — exit
//   /help     — show commands
// Ctrl+C while the model is generating cancels that turn; Ctrl+C at the
// prompt exits.

var opts = Options.Parse(args);
if (opts is null) return 2;

if (!File.Exists(opts.ModelPath))
{
    Console.Error.WriteLine($"Model file not found: {opts.ModelPath}");
    return 1;
}

// In multimodal single-shot mode we always want full native log output —
// the whole point of this harness is to see what mtmd/clip print before
// it explodes. Suppress otherwise.
bool verboseLogs = opts.Verbose || opts.MmprojPath is not null;

// Route native logs to stderr. In verbose mode every line; otherwise only
// warnings and errors so an interactive session isn't drowned.
LlamaBackend.Initialize(logSink: (level, msg) =>
{
    if (verboseLogs || level is LlamaLogLevel.Warn or LlamaLogLevel.Error)
    {
        Console.Error.WriteLine($"[native:{level}] {msg}");
    }
});

Console.Error.WriteLine($"Loading {opts.ModelPath} (gpu_layers={opts.GpuLayers}) ...");
using var model = new LlamaModel(opts.ModelPath, new LlamaModelParameters
{
    GpuLayerCount = opts.GpuLayers,
    UseMmap = true,
});
Console.Error.WriteLine(
    $"Model: layers={model.LayerCount}, n_embd={model.EmbeddingSize}, " +
    $"training_ctx={model.TrainingContextSize}");

using var context = new LlamaContext(model, new LlamaContextParameters
{
    ContextSize = (uint)opts.ContextSize,
    LogicalBatchSize = 512,
    PhysicalBatchSize = 512,
    MaxSequenceCount = 1,
    OffloadKQV = true,
    UseFullSwaCache = true,
});
Console.Error.WriteLine($"Context: n_ctx={context.ContextSize} (requested {opts.ContextSize})");

var template = model.GetChatTemplate();
if (string.IsNullOrEmpty(template))
{
    Console.Error.WriteLine("Warning: model has no embedded chat template; using plain role prefixes.");
}

var sessionCts = new CancellationTokenSource();
CancellationTokenSource? turnCts = null; // mutated in the REPL loop below
// Ctrl+C: if generating, cancel the turn. Otherwise exit cleanly.
Console.CancelKeyPress += (_, ev) =>
{
    if (turnCts is not null && !turnCts.IsCancellationRequested)
    {
        ev.Cancel = true;
        turnCts.Cancel();
        Console.Error.WriteLine("[cancel requested]");
    }
    else
    {
        ev.Cancel = true;
        sessionCts.Cancel();
    }
};

// =====================================================================
// Multimodal single-shot mode. Fires once, prints, exits. Used to
// reproduce crashes in the mtmd path outside of the desktop app where
// stderr survives and we can attach a debugger if needed.
// =====================================================================
if (opts.MmprojPath is not null && opts.ImagePath is not null)
{
    if (!File.Exists(opts.MmprojPath))
    {
        Console.Error.WriteLine($"mmproj file not found: {opts.MmprojPath}");
        return 1;
    }
    if (!File.Exists(opts.ImagePath))
    {
        Console.Error.WriteLine($"image file not found: {opts.ImagePath}");
        return 1;
    }

    Console.Error.WriteLine($"Loading mmproj {opts.MmprojPath} (use_gpu={!opts.CpuEncode}) ...");
    var mtmdParams = new MtmdContextParameters
    {
        UseGpu = !opts.CpuEncode,
        Warmup = true,
    };
    if (opts.ImageMaxTokens is int cap) mtmdParams.ImageMaxTokens = cap;

    using var mtmd = new MtmdContext(model, opts.MmprojPath, mtmdParams);
    Console.Error.WriteLine(
        $"Mtmd: vision={mtmd.SupportsVision}, audio={mtmd.SupportsAudio}, " +
        $"mrope={mtmd.UsesMRope}, non_causal={mtmd.UsesNonCausalMask}, " +
        $"marker='{mtmd.DefaultMediaMarker}'");

    Console.Error.WriteLine($"Loading image {opts.ImagePath} ...");
    using var bitmap = MtmdBitmap.FromFile(mtmd, opts.ImagePath);
    Console.Error.WriteLine($"Bitmap: {bitmap.Width}x{bitmap.Height}, {bitmap.ByteCount} bytes");

    var userText = opts.Prompt ?? "Describe this image.";
    var userContent = $"{mtmd.DefaultMediaMarker}\n{userText}";

    string renderedPrompt = RenderPrompt(template, new[] { new ChatMessage("user", userContent) });
    Console.Error.WriteLine($"Prompt ({renderedPrompt.Length} chars):");
    Console.Error.WriteLine("---");
    Console.Error.WriteLine(renderedPrompt);
    Console.Error.WriteLine("---");

    Console.Error.WriteLine("Prefilling via mtmd_helper_eval_chunks ...");
    var prefillCts = new CancellationTokenSource();
    int newNPast;
    try
    {
        newNPast = await mtmd.EvalPromptAsync(
            context, renderedPrompt, new[] { bitmap },
            nPast: 0, seqId: 0,
            nBatch: (int)context.LogicalBatchSize,
            logitsLast: true, addSpecial: false, parseSpecial: true,
            prefillCts.Token);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"EvalPromptAsync threw: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
    Console.Error.WriteLine($"Prefill done. n_past={newNPast}. Streaming reply ...");

    using var samplerMm = new LlamaSamplerBuilder()
        .WithTopK(40).WithTopP(0.9f).WithMinP(0.05f)
        .WithTemperature(opts.Temperature)
        .WithDistribution(opts.Seed)
        .Build();
    var generatorMm = new LlamaGenerator(context, samplerMm);

    Console.Write("assistant> ");
    await foreach (var piece in generatorMm.StreamFromCurrentStateAsync(
        maxTokens: opts.MaxTokens, cancellationToken: sessionCts.Token))
    {
        Console.Write(piece);
    }
    Console.WriteLine();
    return 0;
}

var history = new List<ChatMessage>();

PrintBanner();

while (!sessionCts.IsCancellationRequested)
{
    Console.Write("user> ");
    var line = Console.ReadLine();
    if (line is null) break;                       // EOF (Ctrl+D)
    line = line.Trim();
    if (line.Length == 0) continue;

    if (line.StartsWith('/'))
    {
        if (!HandleCommand(line, history, context)) break;  // /quit
        continue;
    }

    history.Add(new ChatMessage("user", line));

    // Full-history re-decode each turn: clear KV, rebuild prompt from
    // template, stream the response. Simple and correct; optimising
    // via delta-decode is a later concern.
    context.ClearKvCache();

    string renderedPrompt = RenderPrompt(template, history);

    using var sampler = new LlamaSamplerBuilder()
        .WithPenalties(lastN: 64, repeat: 1.1f)
        .WithTopK(40)
        .WithTopP(0.9f)
        .WithMinP(0.05f)
        .WithTemperature(opts.Temperature)
        .WithDistribution(opts.Seed)
        .Build();

    var generator = new LlamaGenerator(context, sampler);

    Console.Write("assistant> ");
    var assistantSb = new StringBuilder();
    turnCts = CancellationTokenSource.CreateLinkedTokenSource(sessionCts.Token);
    try
    {
        await foreach (var piece in generator.GenerateAsync(
            renderedPrompt,
            maxTokens: opts.MaxTokens,
            addSpecial: false,
            parseSpecial: true,
            cancellationToken: turnCts.Token))
        {
            Console.Write(piece);
            assistantSb.Append(piece);
        }
    }
    catch (OperationCanceledException)
    {
        Console.Write(" [cancelled]");
    }
    finally
    {
        turnCts.Dispose();
        turnCts = null;
        Console.WriteLine();
    }

    var reply = assistantSb.ToString();
    if (reply.Length > 0)
    {
        history.Add(new ChatMessage("assistant", reply));
    }
    else
    {
        // Cancelled before any token — drop the just-added user message so
        // retrying doesn't duplicate it.
        history.RemoveAt(history.Count - 1);
    }
}

Console.Error.WriteLine("bye.");
return 0;

static bool HandleCommand(string line, List<ChatMessage> history, LlamaContext context)
{
    switch (line.Split(' ', 2)[0])
    {
        case "/quit":
        case "/exit":
            return false;
        case "/clear":
            history.Clear();
            context.ClearKvCache();
            Console.Error.WriteLine("[conversation cleared]");
            return true;
        case "/help":
            Console.Error.WriteLine("Commands: /clear, /quit, /help");
            Console.Error.WriteLine("Ctrl+C while generating cancels the current turn.");
            return true;
        default:
            Console.Error.WriteLine($"Unknown command: {line.Split(' ', 2)[0]}. Try /help.");
            return true;
    }
}

static string RenderPrompt(string? template, IReadOnlyList<ChatMessage> history)
{
    if (!string.IsNullOrEmpty(template))
    {
        return LlamaChatTemplate.Apply(template, history, addAssistantPrefix: true);
    }
    var sb = new StringBuilder();
    foreach (var m in history) sb.AppendLine($"{m.Role}: {m.Content}");
    sb.Append("assistant: ");
    return sb.ToString();
}

static void PrintBanner()
{
    Console.Error.WriteLine();
    Console.Error.WriteLine("Ready. Type a message to chat. Commands: /clear /quit /help");
    Console.Error.WriteLine("Ctrl+C while generating cancels; Ctrl+C at the prompt exits.");
    Console.Error.WriteLine();
}

record Options(
    string ModelPath,
    int ContextSize,
    float Temperature,
    uint Seed,
    int MaxTokens,
    int GpuLayers,
    bool Verbose,
    string? MmprojPath,
    string? ImagePath,
    string? Prompt,
    int? ImageMaxTokens,
    bool CpuEncode)
{
    public static Options Default() => new(
        ModelPath: "/mnt/data/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        ContextSize: 2048,
        Temperature: 0.7f,
        Seed: 42,
        MaxTokens: 512,
        GpuLayers: -1,
        Verbose: false,
        MmprojPath: null,
        ImagePath: null,
        Prompt: null,
        ImageMaxTokens: null,
        CpuEncode: false);

    public static Options? Parse(string[] args)
    {
        var o = Default();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string Next(string flag) =>
                (i + 1 < args.Length)
                    ? args[++i]
                    : throw new ArgumentException($"{flag} requires a value");
            switch (a)
            {
                case "--model":       o = o with { ModelPath = Next(a) }; break;
                case "--ctx":         o = o with { ContextSize = int.Parse(Next(a)) }; break;
                case "--temp":        o = o with { Temperature = float.Parse(Next(a)) }; break;
                case "--seed":        o = o with { Seed = uint.Parse(Next(a)) }; break;
                case "--max":         o = o with { MaxTokens = int.Parse(Next(a)) }; break;
                case "--gpu-layers":  o = o with { GpuLayers = int.Parse(Next(a)) }; break;
                case "--verbose":     o = o with { Verbose = true }; break;
                case "--mmproj":      o = o with { MmprojPath = Next(a) }; break;
                case "--image":       o = o with { ImagePath = Next(a) }; break;
                case "--prompt":      o = o with { Prompt = Next(a) }; break;
                case "--image-max-tokens":
                                      o = o with { ImageMaxTokens = int.Parse(Next(a)) }; break;
                case "--cpu-encode":  o = o with { CpuEncode = true }; break;
                case "-h":
                case "--help":
                    Usage();
                    return null;
                default:
                    Console.Error.WriteLine($"unknown flag: {a}");
                    Usage();
                    return null;
            }
        }
        return o;
    }

    static void Usage()
    {
        Console.Error.WriteLine("Usage:");
        Console.Error.WriteLine("  Text REPL:");
        Console.Error.WriteLine("    LlamaChat.Cli [--model PATH] [--ctx N] [--temp F] [--seed N]");
        Console.Error.WriteLine("                  [--max N] [--gpu-layers N] [--verbose]");
        Console.Error.WriteLine("  Multimodal single-shot:");
        Console.Error.WriteLine("    LlamaChat.Cli --model PATH --mmproj PATH --image PATH [--prompt TEXT]");
        Console.Error.WriteLine("                  [--image-max-tokens N] [--cpu-encode]");
        Console.Error.WriteLine("                  [--ctx N] [--gpu-layers N] [--max N] [--seed N]");
    }
}
