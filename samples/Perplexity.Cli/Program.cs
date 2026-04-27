using System.Diagnostics;
using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.Perplexity.Cli;

/// <summary>
/// Headless front-end for <see cref="LlamaPerplexity.ComputeAsync"/>. Same
/// chunked forward-pass scoring as GGUFLab's Perplexity tool.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        var inputs = new List<string>();
        string? corpus = null;
        int contextSize = 512;
        int gpuLayers = -1;
        int threads = -1;
        int batch = 1;
        int concurrency = 1;
        bool scoreSecondHalfOnly = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input"      when i + 1 < args.Length: inputs.Add(args[++i]); break;
                case "--corpus"     when i + 1 < args.Length: corpus = args[++i]; break;
                case "--ctx"        when i + 1 < args.Length: contextSize = int.Parse(args[++i]); break;
                case "--gpu-layers" when i + 1 < args.Length: gpuLayers   = int.Parse(args[++i]); break;
                case "--threads"    when i + 1 < args.Length: threads     = int.Parse(args[++i]); break;
                case "--batch"      when i + 1 < args.Length: batch       = int.Parse(args[++i]); break;
                case "--concurrency" when i + 1 < args.Length: concurrency = int.Parse(args[++i]); break;
                case "--score-all-tokens": scoreSecondHalfOnly = false; break;
                case "-h": case "--help": PrintUsage(); return 0;
                default:
                    Console.Error.WriteLine($"unknown arg: {args[i]}");
                    PrintUsage(); return 2;
            }
        }
        if (inputs.Count == 0 || corpus is null) { PrintUsage(); return 2; }
        if (!File.Exists(corpus)) { Console.Error.WriteLine($"corpus not found: {corpus}"); return 2; }
        foreach (var p in inputs)
            if (!File.Exists(p)) { Console.Error.WriteLine($"input not found: {p}"); return 2; }

        LlamaBackend.Initialize();
        var corpusText = File.ReadAllText(corpus);

        Console.WriteLine($"corpus         {corpus}  ({corpusText.Length:N0} chars)");
        Console.WriteLine($"inputs         {inputs.Count}{(inputs.Count > 1 ? $" (concurrency={concurrency})" : "")}");
        Console.WriteLine($"context size   {contextSize}");
        Console.WriteLine($"batch seqs     {batch}{(batch > 1 ? " (multi-sequence batching)" : " (sequential)")}");
        Console.WriteLine($"score window   {(scoreSecondHalfOnly ? "second half (matches llama.cpp published numbers)" : "all tokens")}");
        Console.WriteLine();

        var pplOpts = new LlamaPerplexityOptions
        {
            ContextSize         = contextSize,
            ScoreSecondHalfOnly = scoreSecondHalfOnly,
            ThreadCount         = threads,
        };
        var modelParams = new LlamaModelParameters
        {
            GpuLayerCount = gpuLayers,
            UseMmap       = true,
        };

        var sw = Stopwatch.StartNew();
        if (inputs.Count == 1 && concurrency == 1)
        {
            // Single-model, in-process: keep the old path so progress
            // reporting stays the legacy "per-chunk" stream.
            using var model = new LlamaModel(inputs[0], modelParams);
            var progress = new Progress<LlamaPerplexityProgress>(p =>
            {
                if (p.ChunkCount > 0)
                    Console.Write($"\rchunk {p.ChunkIndex}/{p.ChunkCount}  running PPL = {p.RunningPerplexity:F4}            ");
            });
            var result = batch > 1
                ? LlamaPerplexity.ComputeBatchedAsync(model, corpusText, pplOpts, batch, progress).GetAwaiter().GetResult()
                : LlamaPerplexity.ComputeAsync(model, corpusText, pplOpts, progress).GetAwaiter().GetResult();
            sw.Stop();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($"perplexity     {result.Perplexity:F4}");
            Console.WriteLine($"mean NLL       {result.NegativeLogLikelihood:F4}");
            Console.WriteLine($"tokens scored  {result.TokensScored:N0}");
            Console.WriteLine($"chunks         {result.ChunkCount}");
            Console.WriteLine($"elapsed        {result.Elapsed.TotalSeconds:F1}s");
            return 0;
        }

        // Multi-input or concurrent mode: use the parallel runner and
        // emit one TSV row per completed job. Result order is "as
        // completed", not input order — caller can sort downstream.
        var jobs = inputs.Select(p => new LlamaPerplexity.PerplexityJob(
            ModelPath:        p,
            Corpus:           corpusText,
            ModelParameters:  modelParams,
            Options:          pplOpts,
            Batch:            batch,
            Tag:              p)).ToList();
        Console.WriteLine($"input\tppl\ttokens\telapsed_s");
        int done = 0;
        await foreach (var jobResult in LlamaPerplexity.RunParallelAsync(jobs, concurrency))
        {
            done++;
            var r = jobResult.Result;
            var name = Path.GetFileName(jobResult.ModelPath);
            Console.WriteLine($"{name}\t{r.Perplexity:F4}\t{r.TokensScored}\t{r.Elapsed.TotalSeconds:F1}");
        }
        sw.Stop();
        Console.WriteLine();
        Console.WriteLine($"# {done} jobs in {sw.Elapsed.TotalSeconds:F1}s wall  (concurrency={concurrency})");
        return 0;
    }

    private static void PrintUsage() => Console.WriteLine("""
        llama-perplexity — chunked-NLL perplexity scoring

        usage:
          llama-perplexity --input <model.gguf> --corpus <text-file> [options]

        options:
          --input <path>            model GGUF (required)
          --corpus <path>           plain-text corpus (required)
          --ctx <n>                 context size (default: 512 — matches the
                                    standard wikitext-2 published-numbers setup)
          --gpu-layers <n>          GPU layer count, -1 = all (default: -1)
          --threads <n>             thread count, -1 = default (default: -1)
          --batch <n>               number of chunks to score in one llama_decode call;
                                    >1 packs that many sequences into a single forward
                                    pass so the GPU isn't underutilized (default: 1).
          --score-all-tokens        score every token instead of only the
                                    second half of each chunk (less standard,
                                    higher PPL — disable to compare against
                                    llama.cpp's published numbers)
        """);
}
