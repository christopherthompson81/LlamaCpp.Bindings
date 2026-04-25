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
    private readonly List<LoadedAdapter> _adapters = new();

    public LlamaModel Model { get; }
    public LlamaContext Context { get; }

    /// <summary>Public name used in OpenAI-style <c>/v1/models</c> responses.</summary>
    public string ModelId { get; }

    /// <summary>
    /// LoRA adapters loaded at startup, exposed so other services that
    /// build their own <see cref="LlamaContext"/> over <see cref="Model"/>
    /// (notably <see cref="DraftHost"/>'s speculative main context) can
    /// mirror the same attachments.
    /// </summary>
    public IReadOnlyList<LoadedAdapter> Adapters => _adapters;

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

        // Resolve device names against the live ggml backend list.
        // Failing fast with the available-device list is a much better
        // operator experience than a silent CUDA0 → wrong-device fallback.
        IReadOnlyList<LlamaComputeDevice>? pinnedDevices = null;
        if (_opts.Devices is { Count: > 0 } deviceNames)
        {
            // EnumerateDevices auto-initialises the backend, but we
            // already called Initialize above (line 40) — calling it
            // here would be a no-op anyway.
            var available = LlamaHardware.EnumerateDevices();
            var picked = new List<LlamaComputeDevice>(deviceNames.Count);
            foreach (var name in deviceNames)
            {
                var dev = available.FirstOrDefault(d =>
                    string.Equals(d.Name, name, StringComparison.OrdinalIgnoreCase));
                if (dev is null)
                {
                    var availList = string.Join(", ", available.Select(d => d.Name));
                    throw new InvalidOperationException(
                        $"LlamaServer:Devices entry '{name}' did not match any registered " +
                        $"compute device. Available: [{availList}].");
                }
                picked.Add(dev);
            }
            pinnedDevices = picked;
        }

        Model = new LlamaModel(_opts.ModelPath, new LlamaModelParameters
        {
            GpuLayerCount  = _opts.GpuLayerCount,
            MainGpu        = _opts.MainGpu,
            SplitMode      = _opts.SplitMode,
            UseMmap        = _opts.UseMmap,
            UseMlock       = _opts.UseMlock,
            CheckTensors   = _opts.CheckTensors,
            UseDirectIo    = _opts.UseDirectIo,
            NoHost         = _opts.NoHost,
            UseExtraBufts  = _opts.UseExtraBufts,
            Devices        = pinnedDevices,
            TensorSplit    = _opts.TensorSplit,
        });

        Context = new LlamaContext(Model, BuildContextParameters(
            _opts, maxSeq: (uint)Math.Max(1, _opts.MaxSequenceCount)));

        ModelId = !string.IsNullOrWhiteSpace(_opts.ModelAlias)
            ? _opts.ModelAlias!
            : Path.GetFileNameWithoutExtension(_opts.ModelPath);

        // LoRA adapters: load each and attach with the configured scale.
        // Bad paths / shape mismatches surface here, at startup, rather
        // than on the first request.
        foreach (var entry in _opts.LoraAdapters)
        {
            if (string.IsNullOrWhiteSpace(entry.Path))
            {
                throw new InvalidOperationException(
                    "LlamaServer:LoraAdapters entry has empty Path.");
            }
            if (!File.Exists(entry.Path))
            {
                throw new FileNotFoundException(
                    $"LoRA adapter file not found: {entry.Path}", entry.Path);
            }
            _log.LogInformation("Loading LoRA adapter {Path} (scale={Scale})", entry.Path, entry.Scale);
            var adapter = LlamaLoraAdapter.LoadFromFile(Model, entry.Path);
            try
            {
                Context.AttachLoraAdapter(adapter, entry.Scale);
            }
            catch
            {
                adapter.Dispose();
                throw;
            }
            _adapters.Add(new LoadedAdapter(adapter, entry.Scale));
        }

        _log.LogInformation("Model loaded. Context sizes: ctx={Ctx}, batch={Batch}, ubatch={Ubatch}, slots={Slots}, adapters={Adapters}",
            Context.ContextSize, Context.LogicalBatchSize, Context.PhysicalBatchSize, Context.MaxSequenceCount, _adapters.Count);
    }

    public void Dispose()
    {
        // Disposal order matters: contexts first (they hold references
        // into adapters), then adapters, then the model. The binding's
        // LoRA docs spell this out — disposing the model before its
        // adapters is undefined behaviour.
        Context.Dispose();
        foreach (var a in _adapters) a.Adapter.Dispose();
        _adapters.Clear();
        Model.Dispose();
    }

    /// <summary>
    /// Build the <see cref="LlamaContextParameters"/> block from operator
    /// configuration. Centralised so the main context and the speculative
    /// main context (in <see cref="DraftHost"/>) share the same field
    /// list — adding a new knob touches one place.
    /// </summary>
    internal static LlamaContextParameters BuildContextParameters(ServerOptions opts, uint maxSeq) =>
        new()
        {
            ContextSize         = (uint)Math.Max(0, opts.ContextSize),
            LogicalBatchSize    = (uint)Math.Max(1, opts.LogicalBatchSize),
            PhysicalBatchSize   = (uint)Math.Max(1, opts.PhysicalBatchSize),
            MaxSequenceCount    = maxSeq,
            OffloadKQV          = opts.OffloadKqv,
            ThreadCount         = opts.ThreadCount,
            BatchThreadCount    = opts.BatchThreadCount,
            FlashAttention      = opts.FlashAttention,
            UseFullSwaCache     = opts.UseFullSwaCache,
            KvCacheTypeK        = opts.KvCacheTypeK,
            KvCacheTypeV        = opts.KvCacheTypeV,
            RopeScalingType     = opts.RopeScalingType,
            RopeFreqBase        = opts.RopeFreqBase,
            RopeFreqScale       = opts.RopeFreqScale,
            YarnExtFactor       = opts.YarnExtFactor,
            YarnAttnFactor      = opts.YarnAttnFactor,
            YarnBetaFast        = opts.YarnBetaFast,
            YarnBetaSlow        = opts.YarnBetaSlow,
            YarnOriginalContext = opts.YarnOriginalContext,
        };
}

/// <summary>
/// A LoRA adapter the server has loaded and attached to <see cref="ModelHost.Context"/>.
/// Disposed in lifecycle order with the model.
/// </summary>
public sealed record LoadedAdapter(LlamaLoraAdapter Adapter, float Scale);
