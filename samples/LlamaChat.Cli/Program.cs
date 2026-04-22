using System.Text;
using LlamaCpp.Bindings;

// Minimal interactive REPL over LlamaCpp.Bindings. Intentionally boring:
// one file, no MVVM, native logs go straight to stderr so we can see what
// llama.cpp is complaining about when things break.
//
// Usage:
//   dotnet run --project samples/LlamaChat.Cli -- [--model PATH] [--ctx N]
//                                                 [--temp F] [--seed N]
//                                                 [--gpu-layers N] [--verbose]
//
// In-session commands:
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

// Route native logs to stderr. In verbose mode every line; otherwise only
// warnings and errors so an interactive session isn't drowned.
LlamaBackend.Initialize(logSink: (level, msg) =>
{
    if (opts.Verbose || level is LlamaLogLevel.Warn or LlamaLogLevel.Error)
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

var history = new List<ChatMessage>();
var sessionCts = new CancellationTokenSource();

// Ctrl+C: if generating, cancel the turn. Otherwise exit cleanly.
CancellationTokenSource? turnCts = null;
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
    bool Verbose)
{
    public static Options Default() => new(
        ModelPath: "/mnt/data/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        ContextSize: 2048,
        Temperature: 0.7f,
        Seed: 42,
        MaxTokens: 512,
        GpuLayers: -1,
        Verbose: false);

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

    static void Usage() => Console.Error.WriteLine(
        "Usage: LlamaChat.Cli [--model PATH] [--ctx N] [--temp F] [--seed N] [--max N] [--gpu-layers N] [--verbose]");
}
