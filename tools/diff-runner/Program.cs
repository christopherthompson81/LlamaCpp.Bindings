// Minimal binding-side runner for the differential test against
// llama-completion. Intentionally narrow surface: load model, build sampler
// per CLI flags, generate tokens, dump them. No chat template, no special
// formatting, no UI cruft. The script that drives both sides does the diff.
//
// Usage:
//   dotnet run --project tools/diff-runner -- \
//     --model PATH --prompt "..." --seed N --max-tokens N \
//     [--temp F] [--top-k N] [--top-p F] [--min-p F] \
//     [--add-special] [--gpu-layers N] [--threads N]
//
// stdout: two lines after a blank line:
//   PROMPT_TOKENS=<comma-separated ids>
//   GEN_TOKENS=<comma-separated ids>
//   GEN_TEXT=<detokenized generation text, single-line, JSON-escaped>
// stderr: native logs, model load info, perf summary.

using System.Globalization;
using System.Text;
using LlamaCpp.Bindings;

var opts = ParseArgs(args);
if (opts is null) return 2;

LlamaBackend.Initialize(logSink: (level, msg) =>
{
    if (opts.Verbose || level is LlamaLogLevel.Warn or LlamaLogLevel.Error)
    {
        Console.Error.WriteLine($"[native:{level}] {msg}");
    }
});

using var model = new LlamaModel(opts.ModelPath, new LlamaModelParameters
{
    GpuLayerCount = opts.GpuLayers,
    UseMmap = true,
});

using var context = new LlamaContext(model, new LlamaContextParameters
{
    ContextSize = 2048,
    LogicalBatchSize = 512,
    PhysicalBatchSize = 512,
    MaxSequenceCount = 1,
    OffloadKQV = true,
    UseFullSwaCache = true,
});

using var sampler = BuildSampler(opts, model);

var promptTokens = model.Vocab.Tokenize(opts.Prompt, addSpecial: opts.AddSpecial, parseSpecial: false);
Console.Error.WriteLine($"prompt tokens: {promptTokens.Length} ids = [{string.Join(',', promptTokens)}]");

// Drive the generator from pre-tokenized input so we can capture both the
// prompt token stream and the generation token stream independently.
var gen = new LlamaGenerator(context, sampler);

var genIds = new List<int>();
var genText = new StringBuilder();
int producedSoFar = 0;

await foreach (var piece in gen.GenerateAsync(promptTokens, maxTokens: opts.MaxTokens, renderSpecialPieces: false))
{
    // Pull the token ids out of the sampler trail. The generator yields decoded
    // pieces but not raw ids; for diff purposes we want the ids too. We get
    // them via vocab.Tokenize(piece) — close enough for ASCII output, and the
    // GEN_TEXT line above is what diff'd byte-for-byte against llama-completion.
    genText.Append(piece);
    producedSoFar++;
}

// Re-tokenize the generated text (without specials) to recover the id stream;
// for greedy decoding this is unambiguous on LLaMA vocabularies. If a future
// run shows divergence here, we know to wire token ids out of the generator
// directly.
genIds = model.Vocab.Tokenize(genText.ToString(), addSpecial: false, parseSpecial: false).ToList();

Console.WriteLine();
Console.WriteLine($"PROMPT_TOKENS={string.Join(',', promptTokens)}");
Console.WriteLine($"GEN_TOKENS={string.Join(',', genIds)}");
Console.WriteLine($"GEN_TEXT={JsonStringEscape(genText.ToString())}");
Console.Error.WriteLine($"generated {producedSoFar} pieces, {genIds.Count} re-tokenized ids");
return 0;

// --------------------------------------------------------------

static LlamaSampler BuildSampler(Options opts, LlamaModel model)
{
    var b = new LlamaSamplerBuilder();
    // Match llama-completion's default chain order: penalties first, then
    // dry, then top-k → top-p → min-p → temperature → terminal. This is the
    // order encoded in llama-completion's --sampler-seq default "edskypmxt".
    if (opts.RepeatPenalty != 1.0f || opts.FrequencyPenalty != 0.0f || opts.PresencePenalty != 0.0f)
    {
        b.WithPenalties(opts.RepeatLastN, opts.RepeatPenalty, opts.FrequencyPenalty, opts.PresencePenalty);
    }
    if (opts.DryMultiplier > 0.0f)
    {
        // llama-completion's defaults: base=1.75, allowed=2, last-n=-1, no breakers.
        b.WithDry(model.Vocab, model.TrainingContextSize,
                  opts.DryMultiplier, opts.DryBase, opts.DryAllowedLength, opts.DryPenaltyLastN);
    }
    if (opts.Temperature == 0.0f)
    {
        // Greedy. llama.cpp's `--temp 0 --top-k 1` is equivalent to greedy;
        // matching here keeps the seed irrelevant for direct comparison.
        return b.WithGreedy().Build();
    }
    if (opts.TopK > 0) b.WithTopK(opts.TopK);
    if (opts.TopP > 0.0f && opts.TopP < 1.0f) b.WithTopP(opts.TopP);
    if (opts.MinP > 0.0f) b.WithMinP(opts.MinP);
    b.WithTemperature(opts.Temperature);
    return b.WithDistribution((uint)opts.Seed).Build();
}

static string JsonStringEscape(string s)
{
    var sb = new StringBuilder(s.Length + 2);
    sb.Append('"');
    foreach (var c in s)
    {
        switch (c)
        {
            case '\\': sb.Append("\\\\"); break;
            case '"':  sb.Append("\\\""); break;
            case '\n': sb.Append("\\n"); break;
            case '\r': sb.Append("\\r"); break;
            case '\t': sb.Append("\\t"); break;
            case '\b': sb.Append("\\b"); break;
            case '\f': sb.Append("\\f"); break;
            default:
                if (c < 0x20)
                    sb.Append($"\\u{(int)c:x4}");
                else
                    sb.Append(c);
                break;
        }
    }
    sb.Append('"');
    return sb.ToString();
}

static Options? ParseArgs(string[] argv)
{
    var o = new Options();
    for (int i = 0; i < argv.Length; i++)
    {
        var a = argv[i];
        string? Next() => i + 1 < argv.Length ? argv[++i] : null;
        switch (a)
        {
            case "--model":       o.ModelPath = Next() ?? ""; break;
            case "--prompt":      o.Prompt    = Next() ?? ""; break;
            case "--seed":        o.Seed      = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--max-tokens":  o.MaxTokens = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--temp":        o.Temperature = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--top-k":       o.TopK      = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--top-p":       o.TopP      = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--min-p":       o.MinP      = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--repeat-penalty":   o.RepeatPenalty   = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--frequency-penalty": o.FrequencyPenalty = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--presence-penalty":  o.PresencePenalty  = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--repeat-last-n":    o.RepeatLastN     = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--dry-multiplier":   o.DryMultiplier   = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--dry-base":         o.DryBase         = float.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--dry-allowed-length": o.DryAllowedLength = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--dry-penalty-last-n": o.DryPenaltyLastN  = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--add-special": o.AddSpecial = true; break;
            case "--gpu-layers":  o.GpuLayers  = int.Parse(Next()!, CultureInfo.InvariantCulture); break;
            case "--verbose":     o.Verbose    = true; break;
            case "-h": case "--help":
                Console.Error.WriteLine("usage: diff-runner --model PATH --prompt STR --seed N --max-tokens N [...]");
                return null;
            default:
                Console.Error.WriteLine($"unknown arg: {a}");
                return null;
        }
    }
    if (string.IsNullOrEmpty(o.ModelPath)) { Console.Error.WriteLine("--model required"); return null; }
    if (o.MaxTokens <= 0) { Console.Error.WriteLine("--max-tokens must be > 0"); return null; }
    return o;
}

internal sealed class Options
{
    public string ModelPath = "";
    public string Prompt = "";
    public int Seed = 42;
    public int MaxTokens = 0;
    public float Temperature = 0.0f;     // 0 = greedy
    public int TopK = 40;
    public float TopP = 0.95f;
    public float MinP = 0.05f;
    public float RepeatPenalty = 1.0f;
    public float FrequencyPenalty = 0.0f;
    public float PresencePenalty = 0.0f;
    public int RepeatLastN = 64;
    public float DryMultiplier = 0.0f;       // 0 = disabled (matches llama-completion default)
    public float DryBase = 1.75f;
    public int DryAllowedLength = 2;
    public int DryPenaltyLastN = -1;         // -1 = full context
    public bool AddSpecial = false;
    public int GpuLayers = 0;            // CPU by default for diff
    public bool Verbose = false;
}
