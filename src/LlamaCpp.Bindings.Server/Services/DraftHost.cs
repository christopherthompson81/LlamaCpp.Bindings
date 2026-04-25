using LlamaCpp.Bindings.Server.Configuration;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Optional singleton wiring up speculative decoding. When
/// <see cref="ServerOptions.DraftModelPath"/> is set, this owns:
/// <list type="bullet">
///   <item>A loaded draft <see cref="LlamaModel"/> + <see cref="LlamaContext"/>.</item>
///   <item>A second <see cref="LlamaContext"/> over the main model — dedicated to
///   speculative requests so its KV state stays out of <see cref="SessionPool"/>.</item>
///   <item>A semaphore that serialises speculative requests (concurrency = 1)
///   because the binding's <c>LlamaSpeculativeGenerator</c> hard-codes
///   sequence id 0 and pollutes KV state across runs.</item>
/// </list>
/// When the path is unset, <see cref="IsAvailable"/> is <c>false</c> and the
/// chat endpoint silently falls back to the non-speculative path even if a
/// caller asks for <c>speculative=true</c>.
/// </summary>
public sealed class DraftHost : IDisposable
{
    private readonly ILogger<DraftHost> _log;
    private readonly LlamaModel? _draftModel;
    private readonly LlamaContext? _draftContext;
    private readonly LlamaContext? _mainContext;
    private readonly SemaphoreSlim? _gate;

    public bool IsAvailable => _draftContext is not null && _mainContext is not null;

    /// <summary>Lookahead this host was configured with. Default 5.</summary>
    public int DraftLookahead { get; }

    /// <summary>Draft model context. Null when speculative is disabled.</summary>
    public LlamaContext? DraftContext => _draftContext;

    /// <summary>
    /// Dedicated main-model context for speculative requests, distinct from
    /// the shared <see cref="ModelHost.Context"/> so its KV state can be
    /// freely reset between runs without disturbing pooled non-speculative
    /// requests. Null when speculative is disabled.
    /// </summary>
    public LlamaContext? MainContext => _mainContext;

    public DraftHost(ModelHost mainHost, IOptions<ServerOptions> options, ILogger<DraftHost> log)
    {
        _log = log;
        var opts = options.Value;
        DraftLookahead = Math.Max(1, opts.DraftLookahead);

        if (string.IsNullOrWhiteSpace(opts.DraftModelPath))
        {
            _log.LogInformation(
                "Speculative decoding disabled: LlamaServer:DraftModelPath not set. " +
                "Chat requests will ignore any speculative=true hint.");
            return;
        }
        if (!File.Exists(opts.DraftModelPath))
        {
            throw new FileNotFoundException(
                $"LlamaServer:DraftModelPath='{opts.DraftModelPath}' does not exist.",
                opts.DraftModelPath);
        }

        LlamaBackend.Initialize();

        _log.LogInformation("Loading draft model from {Path} (gpuLayers={Gpu}, ctx={Ctx})",
            opts.DraftModelPath, opts.DraftGpuLayerCount, opts.DraftContextSize);
        _draftModel = new LlamaModel(opts.DraftModelPath, new LlamaModelParameters
        {
            GpuLayerCount = opts.DraftGpuLayerCount,
            UseMmap       = opts.UseMmap,
            UseMlock      = opts.UseMlock,
        });

        _draftContext = new LlamaContext(_draftModel, new LlamaContextParameters
        {
            ContextSize       = (uint)Math.Max(0, opts.DraftContextSize),
            LogicalBatchSize  = (uint)Math.Max(1, opts.DraftLogicalBatchSize),
            PhysicalBatchSize = (uint)Math.Max(1, opts.DraftPhysicalBatchSize),
            MaxSequenceCount  = 1,
        });

        // Second context over the main model for speculative requests. We
        // can't share ModelHost.Context: the binding's speculative path
        // hard-codes seq=0 and rolls back KV with RemoveSequenceRange,
        // which would clobber any concurrent non-speculative request that
        // also happens to be assigned slot 0. Memory cost is one extra
        // KV-cache footprint on the main model — opt-in only.
        _mainContext = new LlamaContext(mainHost.Model,
            ModelHost.BuildContextParameters(opts, maxSeq: 1));

        // Mirror the main host's LoRA attachments onto the speculative
        // main context — adapters bind to the LlamaModel, which both
        // contexts share, so we can attach the same adapter handle to
        // each context with the same scale.
        foreach (var loaded in mainHost.Adapters)
        {
            _mainContext.AttachLoraAdapter(loaded.Adapter, loaded.Scale);
        }

        _gate = new SemaphoreSlim(1, 1);

        _log.LogInformation(
            "Speculative decoding ready. Draft={DraftPath}, lookahead={Lookahead}, " +
            "main-ctx={MainCtx}, draft-ctx={DraftCtx}",
            opts.DraftModelPath, DraftLookahead,
            _mainContext.ContextSize, _draftContext.ContextSize);
    }

    /// <summary>
    /// Acquire the speculative gate. Returns a disposable handle whose
    /// <c>Dispose</c> resets KV state on both contexts and releases the
    /// permit. Safe to call concurrently — the gate is fair (FIFO).
    /// </summary>
    public async Task<DraftLease> LeaseAsync(CancellationToken ct)
    {
        if (!IsAvailable || _gate is null)
        {
            throw new InvalidOperationException(
                "DraftHost.LeaseAsync called when speculative decoding is not enabled.");
        }
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        return new DraftLease(this);
    }

    internal void ReleaseAfterRequest()
    {
        // Reset KV on both contexts so the next speculative request starts
        // from a clean slate. The binding's docs explicitly call this out
        // for cancelled mid-decode states.
        try { _mainContext?.ClearKvCache(); } catch { /* best-effort */ }
        try { _draftContext?.ClearKvCache(); } catch { /* best-effort */ }
        _gate?.Release();
    }

    public void Dispose()
    {
        _gate?.Dispose();
        _draftContext?.Dispose();
        _mainContext?.Dispose();
        _draftModel?.Dispose();
    }
}

/// <summary>
/// Disposable handle from <see cref="DraftHost.LeaseAsync"/>. Disposing
/// resets KV on both speculative contexts and releases the gate.
/// </summary>
public sealed class DraftLease : IDisposable
{
    private readonly DraftHost _host;
    private bool _disposed;

    internal DraftLease(DraftHost host) => _host = host;

    public LlamaContext MainContext => _host.MainContext!;
    public LlamaContext DraftContext => _host.DraftContext!;
    public int DraftLookahead => _host.DraftLookahead;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _host.ReleaseAfterRequest();
    }
}
