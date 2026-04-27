using System.Diagnostics;
using LlamaCpp.Bindings;

namespace LlamaCpp.Bindings.Imatrix.Cli;

/// <summary>
/// Headless front-end for <see cref="LlamaImatrix.ComputeAsync"/>. Same
/// workflow as GGUFLab's Importance Matrix tool, minus the GUI.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        string? input = null, output = null, corpus = null;
        int contextSize = 512;
        int gpuLayers = -1;
        int threads = -1;
        bool processOutput = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input"   when i + 1 < args.Length: input  = args[++i]; break;
                case "--output"  when i + 1 < args.Length: output = args[++i]; break;
                case "--corpus"  when i + 1 < args.Length: corpus = args[++i]; break;
                case "--ctx"     when i + 1 < args.Length: contextSize = int.Parse(args[++i]); break;
                case "--gpu-layers" when i + 1 < args.Length: gpuLayers = int.Parse(args[++i]); break;
                case "--threads"    when i + 1 < args.Length: threads   = int.Parse(args[++i]); break;
                case "--process-output": processOutput = true; break;
                case "-h": case "--help":
                    PrintUsage(); return 0;
                default:
                    Console.Error.WriteLine($"unknown arg: {args[i]}");
                    PrintUsage(); return 2;
            }
        }
        if (input is null || corpus is null)
        {
            PrintUsage(); return 2;
        }
        output ??= Path.ChangeExtension(input, "imatrix.gguf");

        if (!File.Exists(input))  { Console.Error.WriteLine($"input not found: {input}");   return 2; }
        if (!File.Exists(corpus)) { Console.Error.WriteLine($"corpus not found: {corpus}"); return 2; }

        LlamaBackend.Initialize();
        var corpusText = File.ReadAllText(corpus);
        Console.WriteLine($"input          {input}");
        Console.WriteLine($"corpus         {corpus} ({corpusText.Length:N0} chars)");
        Console.WriteLine($"output         {output}");
        Console.WriteLine($"context size   {contextSize}");
        Console.WriteLine($"gpu layers     {(gpuLayers == -1 ? "all" : gpuLayers.ToString())}");
        Console.WriteLine();

        using var model = new LlamaModel(input, new LlamaModelParameters
        {
            GpuLayerCount = gpuLayers,
            UseMmap       = true,
        });

        var sw = Stopwatch.StartNew();
        var progress = new Progress<LlamaImatrixProgress>(p =>
        {
            if (p.ChunkCount > 0)
                Console.Write($"\rchunk {p.ChunkIndex}/{p.ChunkCount}  tracked={p.TensorsTracked}            ");
        });
        var result = LlamaImatrix.ComputeAsync(
            model, corpusText, output,
            new LlamaImatrixOptions
            {
                ContextSize   = contextSize,
                ProcessOutput = processOutput,
                ThreadCount   = threads,
                DatasetNames  = new[] { Path.GetFileName(corpus) },
            },
            progress).GetAwaiter().GetResult();

        Console.WriteLine();
        Console.WriteLine($"done in        {result.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"chunks         {result.ChunkCount:N0}");
        Console.WriteLine($"tokens         {result.TokensProcessed:N0}");
        Console.WriteLine($"tensors        {result.TensorsTracked:N0}");
        Console.WriteLine($"output         {output}");
        return 0;
    }

    private static void PrintUsage() => Console.WriteLine("""
        llama-imatrix — build an importance matrix from a calibration corpus

        usage:
          llama-imatrix --input <model.gguf> --corpus <text-file> [options]

        options:
          --input <path>        source GGUF (required)
          --corpus <path>       calibration corpus text file (required)
          --output <path>       imatrix output (default: <input>.imatrix.gguf)
          --ctx <n>             context size (default: 512)
          --gpu-layers <n>      GPU layer count, -1 = all (default: -1)
          --threads <n>         thread count, -1 = default (default: -1)
          --process-output      include output.weight in the imatrix
        """);
}
