using System.Diagnostics;
using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.Quantize.Cli;

/// <summary>
/// Headless front-end for <see cref="LlamaQuantizer.QuantizeAsync"/>. Same
/// surface as GGUFLab's Quantize and Adaptive Quantization tools — choose
/// an ftype, optionally attach an imatrix, optionally apply a recipe JSON
/// (whose <c>tt_overrides</c> override per-tensor types).
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        string? input = null, output = null, recipePath = null, imatrixPath = null;
        string? scoresPath = null;
        double tau = 0;
        var ftype = LlamaFileType.Q4_K_M;
        int threads = 0;
        bool allowRequantize = false;
        bool pure = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input"   when i + 1 < args.Length: input  = args[++i]; break;
                case "--output"  when i + 1 < args.Length: output = args[++i]; break;
                case "--ftype"   when i + 1 < args.Length:
                    if (!Enum.TryParse(args[++i], ignoreCase: true, out ftype))
                    { Console.Error.WriteLine($"unknown ftype: {args[i]}"); return 2; }
                    break;
                case "--imatrix" when i + 1 < args.Length: imatrixPath = args[++i]; break;
                case "--recipe"  when i + 1 < args.Length: recipePath  = args[++i]; break;
                case "--recipe-from-scores" when i + 1 < args.Length: scoresPath = args[++i]; break;
                case "--tau"     when i + 1 < args.Length: tau = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                case "--threads" when i + 1 < args.Length: threads     = int.Parse(args[++i]); break;
                case "--allow-requantize": allowRequantize = true; break;
                case "--pure": pure = true; break;
                case "-h": case "--help": PrintUsage(); return 0;
                default:
                    Console.Error.WriteLine($"unknown arg: {args[i]}");
                    PrintUsage(); return 2;
            }
        }
        if (input is null || output is null) { PrintUsage(); return 2; }
        if (!File.Exists(input)) { Console.Error.WriteLine($"input not found: {input}");   return 2; }
        if (imatrixPath is not null && !File.Exists(imatrixPath))
        { Console.Error.WriteLine($"imatrix not found: {imatrixPath}"); return 2; }
        if (recipePath  is not null && !File.Exists(recipePath))
        { Console.Error.WriteLine($"recipe not found: {recipePath}");   return 2; }

        LlamaBackend.Initialize();

        var parameters = new LlamaQuantizationParameters
        {
            FileType        = ftype,
            ThreadCount     = threads,
            AllowRequantize = allowRequantize,
            Pure            = pure,
            ImatrixPath     = imatrixPath,
        };
        if (recipePath is not null && scoresPath is not null)
        {
            Console.Error.WriteLine("--recipe and --recipe-from-scores are mutually exclusive");
            return 2;
        }
        LlamaQuantRecipe? recipe = null;
        if (recipePath is not null)
        {
            recipe = LlamaQuantRecipe.LoadFromJson(recipePath);
        }
        else if (scoresPath is not null)
        {
            if (!(tau > 0))
            {
                Console.Error.WriteLine("--recipe-from-scores requires --tau N (the rel-MSE threshold)");
                return 2;
            }
            var scores = LlamaQuantSensitivity.LoadFromJson(scoresPath);
            recipe = LlamaQuantRecipe.Build(scores, tau, sourceScoreTablePath: scoresPath);
        }
        if (recipe is not null)
        {
            // tt_overrides need !pure so the heuristic still fires for
            // tensors the recipe doesn't cover (1-D norms etc.).
            parameters.Pure = false;
            parameters.TensorTypeOverrides = recipe.ToTtOverrides();
            var src = recipePath ?? $"{scoresPath} @ τ={tau}";
            Console.WriteLine($"recipe         {src}  ({recipe.Entries.Count} per-tensor overrides, avg {recipe.AverageBitsPerElement:F2} bpw)");
        }

        Console.WriteLine($"input          {input}");
        Console.WriteLine($"output         {output}");
        Console.WriteLine($"ftype          {ftype}");
        Console.WriteLine($"imatrix        {imatrixPath ?? "(none)"}");
        Console.WriteLine($"threads        {(threads <= 0 ? "(auto)" : threads.ToString())}");
        Console.WriteLine();

        var sw = Stopwatch.StartNew();
        try
        {
            LlamaQuantizer.QuantizeAsync(input, output, parameters).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"quantize failed: {ex.Message}");
            return 1;
        }
        sw.Stop();

        var inSize  = new FileInfo(input).Length;
        var outSize = new FileInfo(output).Length;
        Console.WriteLine();
        Console.WriteLine($"done in        {sw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"input bytes    {inSize:N0}");
        Console.WriteLine($"output bytes   {outSize:N0}  ({(double)outSize / inSize * 100:F1}% of input)");
        return 0;
    }

    private static void PrintUsage() => Console.WriteLine("""
        llama-quantize — quantize a GGUF, optionally with imatrix and/or recipe

        usage:
          llama-quantize --input <f16.gguf> --output <out.gguf> [options]

        options:
          --input <path>            source GGUF (required)
          --output <path>           destination GGUF (required)
          --ftype <name>            target ftype (default: Q4_K_M). Pass any
                                    LlamaFileType enum name (Q4_0, Q5_K_M, ...).
          --imatrix <path>          imatrix GGUF for imatrix-aware quantization
          --recipe <path>           recipe JSON from a sensitivity sweep — its
                                    tt_overrides take precedence per tensor.
          --threads <n>             worker threads (default: hardware_concurrency)
          --allow-requantize        allow re-quantizing already-quantized tensors
          --pure                    suppress per-tensor heuristic (forces all
                                    tensors to ftype's default; recipe overrides
                                    are ignored under --pure)
        """);
}
