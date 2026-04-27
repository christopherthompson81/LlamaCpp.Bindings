using System.Diagnostics;
using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.SensitivityProfile.Cli;

/// <summary>
/// CLI front-end for <see cref="LlamaSensitivityProfileBuilder.BuildAsync"/>.
/// Builds a per-architecture sensitivity profile in one shot — quantize
/// F16 baseline + per-(category, type) ablations, run all PPL passes
/// through the parallel runner, save profile JSON.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        string? input = null, output = null, corpus = null, imatrix = null, workDir = null;
        int concurrency = 0;  // 0 = auto (LlamaPerplexity.RecommendConcurrency)
        var candidateTypes = new List<LlamaTensorType> { LlamaTensorType.Q2_K, LlamaTensorType.Q4_K, LlamaTensorType.Q6_K };
        List<string>? categories = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input"       when i + 1 < args.Length: input  = args[++i]; break;
                case "--output"      when i + 1 < args.Length: output = args[++i]; break;
                case "--corpus"      when i + 1 < args.Length: corpus = args[++i]; break;
                case "--imatrix"     when i + 1 < args.Length: imatrix = args[++i]; break;
                case "--work-dir"    when i + 1 < args.Length: workDir = args[++i]; break;
                case "--concurrency" when i + 1 < args.Length: concurrency = int.Parse(args[++i]); break;
                case "--candidates"  when i + 1 < args.Length: candidateTypes = ParseCandidates(args[++i]); break;
                case "--categories"  when i + 1 < args.Length:
                    categories = args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                    break;
                case "-h": case "--help": PrintUsage(); return 0;
                default: Console.Error.WriteLine($"unknown arg: {args[i]}"); PrintUsage(); return 2;
            }
        }
        if (input is null || corpus is null) { PrintUsage(); return 2; }
        if (!File.Exists(input))  { Console.Error.WriteLine($"input not found: {input}");   return 2; }
        if (!File.Exists(corpus)) { Console.Error.WriteLine($"corpus not found: {corpus}"); return 2; }
        if (imatrix is not null && !File.Exists(imatrix))
        {
            Console.Error.WriteLine($"imatrix not found: {imatrix}"); return 2;
        }
        if (output is null)
        {
            var stem = Path.GetFileNameWithoutExtension(input);
            var dir  = Path.GetDirectoryName(input) ?? ".";
            output = Path.Combine(dir, $"{stem}.profile.json");
        }

        LlamaBackend.Initialize();

        Console.WriteLine($"input          {input}");
        Console.WriteLine($"corpus         {corpus}");
        Console.WriteLine($"imatrix        {imatrix ?? "(none)"}");
        Console.WriteLine($"output         {output}");
        Console.WriteLine($"candidates     {string.Join(", ", candidateTypes)}");
        Console.WriteLine($"concurrency    {(concurrency > 0 ? concurrency.ToString() : "(auto — VRAM-aware)")}");
        Console.WriteLine();

        var sw = Stopwatch.StartNew();
        var lastStage = LlamaSensitivityProfileBuilder.Stage.Quantizing;
        var progress = new Progress<LlamaSensitivityProfileBuilder.Progress>(p =>
        {
            if (p.Stage != lastStage)
            {
                Console.WriteLine();  // newline between phases
                lastStage = p.Stage;
            }
            Console.Write($"\r[{p.Stage}] {p.CompletedJobs}/{p.TotalJobs}  {p.CurrentLabel ?? ""}                    ");
        });

        var optionsBuilt = new LlamaSensitivityProfileBuilder.Options
        {
            CandidateTypes   = candidateTypes,
            ImatrixPath      = imatrix,
            MaxConcurrent    = concurrency,
            WorkingDirectory = workDir,
        };
        if (categories is not null) optionsBuilt.Categories = categories;
        var profile = await LlamaSensitivityProfileBuilder.BuildAsync(
            sourceModelPath: input,
            corpusPath:      corpus,
            options:         optionsBuilt,
            progress:        progress);
        sw.Stop();
        profile.SaveToJson(output);

        Console.WriteLine();
        Console.WriteLine();
        Console.WriteLine($"=== {profile.ArchitectureId}, {profile.LayerCount} layers ===");
        Console.WriteLine($"baseline F16 PPL    {profile.F16BaselinePerplexity:F4}");
        Console.WriteLine($"build wall          {sw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"profile saved to    {output}");
        Console.WriteLine();
        Console.WriteLine($"{"category",-22} " + string.Join("  ", candidateTypes.Select(t => $"{t,8}")) + "  floor");
        foreach (var cat in profile.CategoriesByDescendingSensitivityAtQ4K)
        {
            var c = profile.Categories[cat];
            var cells = candidateTypes.Select(t => c.DeltaPplByType.TryGetValue(t, out var v) ? $"{v,8:+0.0000;-0.0000}" : $"{"--",8}");
            var floorStr = c.RecommendedFloor?.ToString() ?? "—";
            Console.WriteLine($"{cat,-22} " + string.Join("  ", cells) + $"  {floorStr}");
        }
        return 0;
    }

    private static List<LlamaTensorType> ParseCandidates(string csv) =>
        csv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
           .Select(t => Enum.Parse<LlamaTensorType>(t, ignoreCase: true))
           .ToList();

    private static void PrintUsage() => Console.WriteLine("""
        llama-sensitivity-profile — build a per-architecture sensitivity profile

        usage:
          llama-sensitivity-profile --input <model.gguf> --corpus <corpus.txt> [options]

        For each requested candidate type and each weight category, quantize
        ONLY that category to the test type (everything else stays F16) and
        score wikitext PPL. ΔPPL vs F16 baseline forms the per-(category, type)
        coefficient. Catastrophic per-category drops set a "do not go below"
        floor for the recipe builder.

        options:
          --input <path>         source GGUF (F16 / BF16) (required)
          --corpus <path>        plain-text calibration corpus (required)
          --imatrix <path>       imatrix GGUF for imatrix-aware quantization
          --output <path>        profile JSON destination (default: alongside input)
          --candidates <csv>     candidate types to ablate at (default: Q2_K,Q4_K,Q6_K)
          --categories <csv>     weight categories to score (default: 7 standard
                                 transformer cats; expand with output.weight,
                                 token_embd.weight for per-tensor coverage)
          --concurrency <n>      concurrent PPL jobs (default: auto — picks based on
                                 file sizes vs available VRAM, capped at half the
                                 logical CPU cores). Pass an explicit value to override.
          --work-dir <path>      where to write temp ablation GGUFs (default: a fresh
                                 dir under /tmp). Disk peak is concurrency × source
                                 model size — pass a path with at least 30 GB free
                                 for 1.7B-class models.
        """);
}
