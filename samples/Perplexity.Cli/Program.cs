using System.Diagnostics;
using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.Perplexity.Cli;

/// <summary>
/// Headless front-end for <see cref="LlamaPerplexity.ComputeAsync"/>. Same
/// chunked forward-pass scoring as GGUFLab's Perplexity tool.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        string? input = null, corpus = null;
        int contextSize = 512;
        int gpuLayers = -1;
        int threads = -1;
        bool scoreSecondHalfOnly = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input"      when i + 1 < args.Length: input  = args[++i]; break;
                case "--corpus"     when i + 1 < args.Length: corpus = args[++i]; break;
                case "--ctx"        when i + 1 < args.Length: contextSize = int.Parse(args[++i]); break;
                case "--gpu-layers" when i + 1 < args.Length: gpuLayers   = int.Parse(args[++i]); break;
                case "--threads"    when i + 1 < args.Length: threads     = int.Parse(args[++i]); break;
                case "--score-all-tokens": scoreSecondHalfOnly = false; break;
                case "-h": case "--help": PrintUsage(); return 0;
                default:
                    Console.Error.WriteLine($"unknown arg: {args[i]}");
                    PrintUsage(); return 2;
            }
        }
        if (input is null || corpus is null) { PrintUsage(); return 2; }
        if (!File.Exists(input))  { Console.Error.WriteLine($"input not found: {input}");   return 2; }
        if (!File.Exists(corpus)) { Console.Error.WriteLine($"corpus not found: {corpus}"); return 2; }

        LlamaBackend.Initialize();
        var corpusText = File.ReadAllText(corpus);

        Console.WriteLine($"input          {input}  ({new FileInfo(input).Length / 1024 / 1024} MB)");
        Console.WriteLine($"corpus         {corpus}  ({corpusText.Length:N0} chars)");
        Console.WriteLine($"context size   {contextSize}");
        Console.WriteLine($"score window   {(scoreSecondHalfOnly ? "second half (matches llama.cpp published numbers)" : "all tokens")}");
        Console.WriteLine();

        using var model = new LlamaModel(input, new LlamaModelParameters
        {
            GpuLayerCount = gpuLayers,
            UseMmap       = true,
        });

        var sw = Stopwatch.StartNew();
        var progress = new Progress<LlamaPerplexityProgress>(p =>
        {
            if (p.ChunkCount > 0)
                Console.Write($"\rchunk {p.ChunkIndex}/{p.ChunkCount}  running PPL = {p.RunningPerplexity:F4}            ");
        });
        var result = LlamaPerplexity.ComputeAsync(
            model, corpusText,
            new LlamaPerplexityOptions
            {
                ContextSize         = contextSize,
                ScoreSecondHalfOnly = scoreSecondHalfOnly,
                ThreadCount         = threads,
            },
            progress).GetAwaiter().GetResult();
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
          --score-all-tokens        score every token instead of only the
                                    second half of each chunk (less standard,
                                    higher PPL — disable to compare against
                                    llama.cpp's published numbers)
        """);
}
