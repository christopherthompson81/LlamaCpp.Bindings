using LlamaCpp.Bindings.Server.Configuration;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Singleton that owns the <see cref="LlamaModel"/> and the shared
/// <see cref="LlamaContext"/> for the server's lifetime. Everything below
/// it (session pool, endpoints) borrows through this type — there is
/// exactly one model per process in V1.
/// </summary>
public sealed class ModelHost : IDisposable
{
    private readonly ILogger<ModelHost> _log;
    private readonly ServerOptions _opts;

    public LlamaModel Model { get; }
    public LlamaContext Context { get; }

    /// <summary>Public name used in OpenAI-style <c>/v1/models</c> responses.</summary>
    public string ModelId { get; }

    public ModelHost(IOptions<ServerOptions> options, ILogger<ModelHost> log)
    {
        _log = log;
        _opts = options.Value;

        if (string.IsNullOrWhiteSpace(_opts.ModelPath))
        {
            throw new InvalidOperationException(
                "LlamaServer:ModelPath is not set. Specify a GGUF path in appsettings.json or " +
                "via --LlamaServer:ModelPath=/path/to/model.gguf on the command line.");
        }
        if (!File.Exists(_opts.ModelPath))
        {
            throw new FileNotFoundException(
                $"Model file not found: {_opts.ModelPath}", _opts.ModelPath);
        }

        LlamaBackend.Initialize();

        _log.LogInformation("Loading model from {Path} (gpuLayers={Gpu}, ctx={Ctx}, slots={Slots})",
            _opts.ModelPath, _opts.GpuLayerCount, _opts.ContextSize, _opts.MaxSequenceCount);

        Model = new LlamaModel(_opts.ModelPath, new LlamaModelParameters
        {
            GpuLayerCount = _opts.GpuLayerCount,
            MainGpu       = _opts.MainGpu,
            SplitMode     = _opts.SplitMode,
            UseMmap       = _opts.UseMmap,
            UseMlock      = _opts.UseMlock,
            CheckTensors  = _opts.CheckTensors,
        });

        Context = new LlamaContext(Model, new LlamaContextParameters
        {
            ContextSize       = (uint)Math.Max(0, _opts.ContextSize),
            LogicalBatchSize  = (uint)Math.Max(1, _opts.LogicalBatchSize),
            PhysicalBatchSize = (uint)Math.Max(1, _opts.PhysicalBatchSize),
            MaxSequenceCount  = (uint)Math.Max(1, _opts.MaxSequenceCount),
            OffloadKQV        = _opts.OffloadKqv,
            ThreadCount       = _opts.ThreadCount,
            BatchThreadCount  = _opts.BatchThreadCount,
            FlashAttention    = _opts.FlashAttention,
            UseFullSwaCache   = _opts.UseFullSwaCache,
            KvCacheTypeK      = _opts.KvCacheTypeK,
            KvCacheTypeV      = _opts.KvCacheTypeV,
        });

        ModelId = !string.IsNullOrWhiteSpace(_opts.ModelAlias)
            ? _opts.ModelAlias!
            : Path.GetFileNameWithoutExtension(_opts.ModelPath);

        _log.LogInformation("Model loaded. Context sizes: ctx={Ctx}, batch={Batch}, ubatch={Ubatch}, slots={Slots}",
            Context.ContextSize, Context.LogicalBatchSize, Context.PhysicalBatchSize, Context.MaxSequenceCount);
    }

    public void Dispose()
    {
        Context.Dispose();
        Model.Dispose();
    }
}
