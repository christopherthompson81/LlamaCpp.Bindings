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

        // NUMA placement is process-wide; configure it before the first
        // model load so allocator behaviour is consistent across every
        // host (main + embedding + rerank + draft). No-op on non-NUMA
        // hardware and when the strategy is Disabled (the default).
        if (_opts.NumaStrategy != LlamaNumaStrategy.Disabled)
        {
            _log.LogInformation("Initialising NUMA strategy: {Strategy}", _opts.NumaStrategy);
            LlamaBackend.InitializeNuma(_opts.NumaStrategy);
        }

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

        // Resolve tensor-buft overrides + the optional --cpu-moe preset.
        // Both share the same plumbing — the preset just appends a
        // canonical pattern that routes MoE FFN tensors to the CPU buft.
        var availableForOverrides = (pinnedDevices ?? (IReadOnlyList<LlamaComputeDevice>?)null)
            ?? LlamaHardware.EnumerateDevices();
        var overrides = ResolveTensorBuftOverrides(_opts, availableForOverrides);

        Model = new LlamaModel(_opts.ModelPath, new LlamaModelParameters
        {
            GpuLayerCount       = _opts.GpuLayerCount,
            MainGpu             = _opts.MainGpu,
            SplitMode           = _opts.SplitMode,
            UseMmap             = _opts.UseMmap,
            UseMlock            = _opts.UseMlock,
            CheckTensors        = _opts.CheckTensors,
            UseDirectIo         = _opts.UseDirectIo,
            NoHost              = _opts.NoHost,
            UseExtraBufts       = _opts.UseExtraBufts,
            Devices             = pinnedDevices,
            TensorSplit         = _opts.TensorSplit,
            TensorBuftOverrides = overrides.Count > 0 ? overrides : null,
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

        // Control vectors: load each, merge by element-wise sum (with
        // the configured per-file scale already baked in), then attach
        // to the shared context with the optional layer range.
        ActiveControlVector = ResolveAndAttachControlVector(_opts, Context, _log);

        _log.LogInformation("Model loaded. Context sizes: ctx={Ctx}, batch={Batch}, ubatch={Ubatch}, slots={Slots}, adapters={Adapters}, cvec={Cvec}",
            Context.ContextSize, Context.LogicalBatchSize, Context.PhysicalBatchSize, Context.MaxSequenceCount, _adapters.Count,
            ActiveControlVector is not null ? "yes" : "no");
    }

    /// <summary>
    /// Merged control vector applied to the shared context, exposed so
    /// <see cref="DraftHost"/> can mirror the same data onto its
    /// dedicated speculative main context.
    /// </summary>
    public LlamaControlVector? ActiveControlVector { get; }

    /// <summary>The layer range used when attaching <see cref="ActiveControlVector"/>. Null when no vector is active.</summary>
    public (int Start, int End)? ActiveControlVectorRange { get; private set; }

    /// <summary>
    /// Load + merge + attach the operator-supplied control vectors.
    /// Returns the merged vector (so callers can re-attach it to other
    /// contexts), or null when none were configured. Bad paths /
    /// dimension mismatches surface eagerly via <see cref="LlamaControlVector.LoadFromFile"/>.
    /// </summary>
    private LlamaControlVector? ResolveAndAttachControlVector(
        ServerOptions opts, LlamaContext context, ILogger log)
    {
        if (opts.ControlVectors.Count == 0) return null;

        LlamaControlVector? merged = null;
        foreach (var entry in opts.ControlVectors)
        {
            if (string.IsNullOrWhiteSpace(entry.Path))
            {
                throw new InvalidOperationException(
                    "LlamaServer:ControlVectors entry has empty Path.");
            }
            if (!File.Exists(entry.Path))
            {
                throw new FileNotFoundException(
                    $"Control-vector file not found: {entry.Path}", entry.Path);
            }
            log.LogInformation("Loading control vector {Path} (scale={Scale})", entry.Path, entry.Scale);
            var loaded = LlamaControlVector.LoadFromFile(entry.Path, entry.Scale);
            merged = merged is null ? loaded : merged.Combine(loaded);
        }
        if (merged is null) return null;

        int start = opts.ControlVectorLayerStart ?? 1;
        int end = opts.ControlVectorLayerEnd ?? merged.LayerCount;
        context.SetControlVector(merged, start, end);
        ActiveControlVectorRange = (start, end);
        log.LogInformation(
            "Control vector applied: n_embd={NEmbd}, layers={Layers}, range=[{Start}..{End}]",
            merged.NEmbd, merged.LayerCount, start, end);
        return merged;
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
    /// llama.cpp's canonical regex matching MoE expert FFN tensors —
    /// <c>blk.*.ffn_(up|down|gate|gate_up)_(ch|)exps</c>. Used by
    /// <see cref="ServerOptions.CpuMoe"/> to route those tensors to CPU
    /// memory, keeping the dense layers on GPU.
    /// </summary>
    private const string CpuMoeRegex = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

    /// <summary>
    /// Resolve operator-supplied tensor-buft overrides + the optional
    /// <see cref="ServerOptions.CpuMoe"/> preset to a list the binding
    /// can pin into the load call. Bad device names fail fast with the
    /// available list spelled out.
    /// </summary>
    private static List<LlamaTensorBuftOverride> ResolveTensorBuftOverrides(
        ServerOptions opts, IReadOnlyList<LlamaComputeDevice> available)
    {
        var result = new List<LlamaTensorBuftOverride>();

        foreach (var entry in opts.TensorBuftOverrides)
        {
            if (string.IsNullOrEmpty(entry.Pattern))
            {
                throw new InvalidOperationException(
                    "LlamaServer:TensorBuftOverrides entry has empty Pattern.");
            }
            if (string.IsNullOrEmpty(entry.Device))
            {
                throw new InvalidOperationException(
                    $"LlamaServer:TensorBuftOverrides entry for pattern '{entry.Pattern}' has empty Device.");
            }
            var device = available.FirstOrDefault(d =>
                string.Equals(d.Name, entry.Device, StringComparison.OrdinalIgnoreCase));
            if (device is null)
            {
                var availList = string.Join(", ", available.Select(d => d.Name));
                throw new InvalidOperationException(
                    $"LlamaServer:TensorBuftOverrides device '{entry.Device}' did not match " +
                    $"any registered compute device. Available: [{availList}].");
            }
            var buft = entry.Host
                ? LlamaBufferType.HostFrom(device)
                : LlamaBufferType.From(device);
            if (buft is null)
            {
                throw new InvalidOperationException(
                    $"Device '{entry.Device}' has no host-pinned buffer type — set Host=false " +
                    "or pick a different device.");
            }
            result.Add(new LlamaTensorBuftOverride(entry.Pattern, buft));
        }

        if (opts.CpuMoe)
        {
            // The preset always routes to the CPU device's buft. Walk
            // every registered device since CPU isn't necessarily in
            // the operator's pinned-devices list.
            var allDevices = LlamaHardware.EnumerateDevices();
            var cpuDevice = allDevices.FirstOrDefault(d => d.Type == LlamaComputeDeviceType.Cpu);
            if (cpuDevice is null)
            {
                throw new InvalidOperationException(
                    "LlamaServer:CpuMoe is set but no CPU compute device was registered. " +
                    "Backend plugins may have failed to load.");
            }
            result.Add(new LlamaTensorBuftOverride(CpuMoeRegex, LlamaBufferType.From(cpuDevice)));
        }

        return result;
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
