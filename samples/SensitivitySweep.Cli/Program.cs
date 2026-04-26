using System.Collections.Concurrent;
using System.Diagnostics;
using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.SensitivitySweep.Cli;

/// <summary>
/// CLI front-end for <see cref="LlamaQuantSensitivity.MeasureAsync"/>. Same workflow as
/// GGUFLab's Adaptive Quantization page, minus the GUI — useful for headless runs,
/// reproducible benchmarks, and CI smoke tests.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        var opts = ParseArgs(args);
        if (opts is null) { PrintUsage(); return 2; }

        if (!File.Exists(opts.InputPath))
        {
            Console.Error.WriteLine($"input not found: {opts.InputPath}");
            return 2;
        }
        if (opts.ImatrixPath is not null && !File.Exists(opts.ImatrixPath))
        {
            Console.Error.WriteLine($"imatrix not found: {opts.ImatrixPath}");
            return 2;
        }

        LlamaBackend.Initialize();

        var measureOpts = new LlamaQuantSensitivityOptions
        {
            ImatrixPath            = opts.ImatrixPath,
            IncludeNameRegex       = opts.IncludeNameRegex,
            MaxDegreeOfParallelism = opts.MaxParallel ?? 0,
        };
        if (opts.Candidates is { } cands)
        {
            measureOpts.CandidateTypes = cands;
        }

        // Phase-timing accumulators. The progress callback fires per phase
        // boundary, so summing the deltas gives a candidate-by-candidate
        // wallclock breakdown — close-to-accurate when Parallel.For is
        // disabled (--max-parallel 1), illustrative when it isn't.
        var phaseTotals      = new ConcurrentDictionary<LlamaQuantSensitivityPhase, double>();
        var lastPhaseTickUtc = new ConcurrentDictionary<int, long>();   // candidate ix → ticks
        var lastPhaseKind    = new ConcurrentDictionary<int, LlamaQuantSensitivityPhase>();
        long sourceDequantStartTicks = 0, sourceDequantTotalTicks = 0;
        int  tensorCount = 0;
        int  candidatesCount = 0;
        int  lastPrintedTensor = -1;

        var sw = Stopwatch.StartNew();
        var progress = new Progress<LlamaQuantSensitivityProgress>(p =>
        {
            tensorCount     = p.TensorCount;
            candidatesCount = p.CandidatesPerTensor;
            long now = Stopwatch.GetTimestamp();
            switch (p.Phase)
            {
                case LlamaQuantSensitivityPhase.Tensor:
                    if (opts.Verbose && p.TensorIndex != lastPrintedTensor)
                    {
                        Console.WriteLine($"[{Wall(sw)}] tensor {p.TensorIndex}/{p.TensorCount}: {p.CurrentTensorName}");
                        lastPrintedTensor = p.TensorIndex;
                    }
                    lastPhaseTickUtc.Clear();
                    lastPhaseKind.Clear();
                    break;
                case LlamaQuantSensitivityPhase.SourceDequantize:
                    sourceDequantStartTicks = now;
                    break;
                case LlamaQuantSensitivityPhase.Quantize:
                case LlamaQuantSensitivityPhase.Dequantize:
                case LlamaQuantSensitivityPhase.Score:
                    if (lastPhaseTickUtc.TryGetValue(p.CandidateIndex, out var t0)
                        && lastPhaseKind.TryGetValue(p.CandidateIndex, out var k0))
                    {
                        Add(phaseTotals, k0, ElapsedSeconds(t0, now));
                    }
                    else if (sourceDequantStartTicks > 0)
                    {
                        // First candidate phase tick after source-dequant
                        // start — close the source-dequant span.
                        Interlocked.Add(ref sourceDequantTotalTicks, now - sourceDequantStartTicks);
                        sourceDequantStartTicks = 0;
                    }
                    lastPhaseTickUtc[p.CandidateIndex] = now;
                    lastPhaseKind[p.CandidateIndex]    = p.Phase;
                    break;
                case LlamaQuantSensitivityPhase.CandidateDone:
                    if (lastPhaseTickUtc.TryGetValue(p.CandidateIndex, out var t1)
                        && lastPhaseKind.TryGetValue(p.CandidateIndex, out var k1))
                    {
                        Add(phaseTotals, k1, ElapsedSeconds(t1, now));
                    }
                    lastPhaseTickUtc.TryRemove(p.CandidateIndex, out _);
                    lastPhaseKind.TryRemove(p.CandidateIndex, out _);
                    if (opts.Verbose)
                    {
                        var mse = p.CandidateRelativeMse?.ToString("E2") ?? "—";
                        Console.WriteLine($"  [{Wall(sw)}] {p.CandidateType,-6} ✓  rel-MSE {mse}");
                    }
                    break;
            }
        });

        Console.WriteLine($"input          {opts.InputPath}");
        Console.WriteLine($"imatrix        {opts.ImatrixPath ?? "(none)"}");
        Console.WriteLine($"include regex  {opts.IncludeNameRegex ?? "(none)"}");
        Console.WriteLine($"max parallel   {(opts.MaxParallel?.ToString() ?? "(auto)")}");
        Console.WriteLine();

        LlamaQuantSensitivityResult result;
        try
        {
            result = LlamaQuantSensitivity.MeasureAsync(opts.InputPath, measureOpts, progress).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"sweep failed: {ex.Message}");
            return 1;
        }
        sw.Stop();

        var outPath = opts.OutputPath
            ?? Path.Combine(Path.GetDirectoryName(opts.InputPath) ?? ".",
                            Path.GetFileNameWithoutExtension(opts.InputPath) + ".scores.json");
        LlamaQuantSensitivity.SaveToJson(result, outPath);

        Console.WriteLine();
        Console.WriteLine($"swept          {result.Scores.Count} rows ({tensorCount} tensors × {candidatesCount} candidates)");
        Console.WriteLine($"wall clock     {result.Elapsed.TotalSeconds:F2} s");
        Console.WriteLine($"output         {outPath}");

        if (opts.Benchmark)
        {
            // Sum across all tensors & candidates. With parallelism > 1
            // these are CPU-time totals (sum of every concurrent worker's
            // wall span), which is what you want for relative weighting.
            // The wall-clock above is the ground truth for "did we
            // actually finish faster".
            double totalCpu =
                (sourceDequantTotalTicks / (double)Stopwatch.Frequency) +
                phaseTotals.Values.Sum();
            Console.WriteLine();
            Console.WriteLine("phase breakdown (CPU-time across all tensors & candidates)");
            Console.WriteLine($"  {"source dequantize",-22}  {sourceDequantTotalTicks / (double)Stopwatch.Frequency,8:F2} s   {Pct(sourceDequantTotalTicks / (double)Stopwatch.Frequency, totalCpu)}");
            foreach (var phase in new[] { LlamaQuantSensitivityPhase.Quantize,
                                          LlamaQuantSensitivityPhase.Dequantize,
                                          LlamaQuantSensitivityPhase.Score })
            {
                double t = phaseTotals.TryGetValue(phase, out var v) ? v : 0;
                Console.WriteLine($"  {phase,-22}  {t,8:F2} s   {Pct(t, totalCpu)}");
            }
            Console.WriteLine($"  {"total CPU",-22}  {totalCpu,8:F2} s");
            Console.WriteLine($"  {"speedup vs serial",-22}  {totalCpu / result.Elapsed.TotalSeconds,8:F2}x");
        }
        return 0;
    }

    private static double ElapsedSeconds(long t0, long t1) =>
        (t1 - t0) / (double)Stopwatch.Frequency;

    private static void Add(ConcurrentDictionary<LlamaQuantSensitivityPhase, double> d,
                            LlamaQuantSensitivityPhase k, double v)
    {
        d.AddOrUpdate(k, v, (_, old) => old + v);
    }

    private static string Pct(double t, double total) =>
        total > 0 ? $"({t / total * 100,5:F1}%)" : "";

    private static string Wall(Stopwatch s) =>
        $"{s.Elapsed.TotalSeconds,7:F2}s";

    private sealed class Options
    {
        public string InputPath = "";
        public string? OutputPath;
        public string? ImatrixPath;
        public string? IncludeNameRegex;
        public int? MaxParallel;
        public IReadOnlyList<LlamaTensorType>? Candidates;
        public bool Verbose;
        public bool Benchmark;
    }

    private static Options? ParseArgs(string[] args)
    {
        var opts = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input"        when i + 1 < args.Length: opts.InputPath = args[++i]; break;
                case "--output"       when i + 1 < args.Length: opts.OutputPath = args[++i]; break;
                case "--imatrix"      when i + 1 < args.Length: opts.ImatrixPath = args[++i]; break;
                case "--include"      when i + 1 < args.Length: opts.IncludeNameRegex = args[++i]; break;
                case "--max-parallel" when i + 1 < args.Length: opts.MaxParallel = int.Parse(args[++i]); break;
                case "--candidates"   when i + 1 < args.Length: opts.Candidates = ParseCandidates(args[++i]); break;
                case "-v":
                case "--verbose":     opts.Verbose = true; break;
                case "--benchmark":   opts.Benchmark = true; break;
                case "-h":
                case "--help":        return null;
                default:
                    Console.Error.WriteLine($"unknown arg: {args[i]}");
                    return null;
            }
        }
        if (string.IsNullOrEmpty(opts.InputPath)) return null;
        return opts;
    }

    private static IReadOnlyList<LlamaTensorType> ParseCandidates(string csv)
    {
        var list = new List<LlamaTensorType>();
        foreach (var token in csv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            list.Add(Enum.Parse<LlamaTensorType>(token, ignoreCase: true));
        }
        return list;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
            llama-sensitivity-sweep — per-tensor quantization sensitivity sweep

            usage:
              llama-sensitivity-sweep --input <model.gguf> [options]

            options:
              --input <path>            source GGUF (required)
              --output <path>           write scores JSON here (default: <input>.scores.json)
              --imatrix <path>          optional imatrix GGUF for column weighting
              --include <regex>         only score tensors whose name matches
              --max-parallel <n>        cap concurrent candidates per tensor (default: physical cores)
              --candidates <csv>        e.g. F16,Q8_0,Q6_K,Q4_K (default: full ladder of 11 types)
              --verbose, -v             stream per-tensor / per-candidate progress
              --benchmark               print phase-time breakdown after the sweep

            example:
              llama-sensitivity-sweep \
                --input ~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.gguf \
                --candidates Q8_0,Q6_K,Q4_K,Q3_K,IQ4_XS,IQ3_S,IQ2_S \
                --max-parallel 8 \
                --benchmark
            """);
    }
}
