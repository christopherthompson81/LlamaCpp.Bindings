using System;
using System.Collections.Generic;
using System.IO;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Produces a recommended <see cref="ModelLoadSettings"/> /
/// <see cref="SamplerSettings"/> bundle for a given GGUF.
///
/// The planner runs in two phases:
/// <list type="number">
/// <item>
///     <b>Metadata probe</b> (gpu_layers=0, mmap): load the model briefly
///     to read architecture, training ctx, size, layer count. Used to pick
///     the sampling profile and the ideal-plan candidates. Cheap — mmap
///     doesn't pull the weights into RAM, much less VRAM.
/// </item>
/// <item>
///     <b>Empirical load probe</b> (gpu_layers=-1, mmap): actually load
///     the model onto the GPU and try to create a context at the desired
///     ctx / KV config. If that fails, step the ctx down and retry against
///     the same loaded model. First successful config wins. The model and
///     context are disposed — llama.cpp's CUDA allocator cleanly returns
///     memory to the OS, so the user's subsequent real load has full VRAM
///     available.
/// </item>
/// </list>
///
/// Empirical probing replaces paper-math estimation of KV-per-token —
/// formula-based math underestimates for hybrid architectures like
/// qwen35moe (gated delta net) where the "KV cache" is a compact recurrent
/// state, not the O(ctx × layers × heads × head_dim) standard transformer
/// KV. Such models load 262K contexts in a fraction of the memory our
/// formula would reserve, and guessing wrong costs users thousands of
/// usable context tokens.
/// </summary>
public static class AutoConfigureService
{
    // Context-size candidates expressed as fractions of training ctx.
    // Tried in order; first one that loads wins. The 1.0 entry is the
    // "ideal" — users with enough hardware land here and never see the
    // others get exercised.
    private static readonly double[] CtxCandidateFractions = { 1.0, 0.75, 0.5, 0.25 };

    // Floor / ceiling on context size. Above the ceiling most sampling
    // profiles stop being useful regardless; below the floor chat is too
    // short to be interactive.
    private const uint MinContextSize = 4096;
    private const uint MaxContextSize = 1_048_576;

    /// <summary>
    /// Run the full probe + inference pipeline for <paramref name="modelPath"/>.
    /// </summary>
    /// <param name="modelPath">Path to a GGUF model file.</param>
    /// <param name="samplingDb">
    /// Optional pre-loaded sampling database. Pass <c>null</c> (the default)
    /// to use the embedded Avalonia asset. Tests use this seam to supply a
    /// database loaded from disk without standing up an Avalonia runtime.
    /// </param>
    /// <exception cref="FileNotFoundException">The GGUF file doesn't exist.</exception>
    /// <exception cref="LlamaException">Native model load failed.</exception>
    public static AutoConfigureResult Configure(
        string modelPath, SamplingProfileDatabase? samplingDb = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Model file not found.", modelPath);

        // LlamaBackend.Initialize is idempotent; safe to call even if the
        // app has already initialised (ChatSession.Load does so on real
        // model load). First-call-wins on the log-sink registration.
        LlamaBackend.Initialize();

        // --- Phase 1: metadata probe ---
        string? archName, humanName;
        int trainCtx;
        long modelBytes;
        int nLayers;
        using (var metaModel = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
            UseMlock = false,
        }))
        {
            var meta = metaModel.Metadata;
            meta.TryGetValue("general.architecture", out archName);
            meta.TryGetValue("general.name", out humanName);
            trainCtx    = Math.Max((int)MinContextSize, metaModel.TrainingContextSize);
            modelBytes  = metaModel.SizeInBytes;
            nLayers     = Math.Max(1, metaModel.LayerCount);
        }

        var devices = LlamaHardware.EnumerateDevices();
        long freeVramBytes = 0;
        foreach (var d in devices)
        {
            if (d.Type is LlamaComputeDeviceType.Gpu or LlamaComputeDeviceType.IntegratedGpu)
                freeVramBytes += d.FreeBytes;
        }
        bool gpuAvailable = freeVramBytes > 0 && LlamaBackend.SupportsGpuOffload();

        // --- Sampling profile lookup (independent of the load probe) ---
        var samplerEntry = samplingDb is not null
            ? SamplingProfileDb.Match(samplingDb, archName, humanName)
            : SamplingProfileDb.Match(archName, humanName);
        var sampler = ApplyProfile(SamplerSettings.Default, samplerEntry.Sampling);

        // --- Phase 2: empirical load probe ---
        uint targetCtx = (uint)Math.Clamp(trainCtx, (int)MinContextSize, (int)MaxContextSize);
        var candidates = BuildCandidates(targetCtx);

        ProbeResult? winner = null;
        string probeNotes;
        if (!gpuAvailable)
        {
            // No GPU — skip load probing, recommend CPU-only at full ctx
            // and let the user start the actual load from there.
            winner = new ProbeResult(
                Ctx: targetCtx,
                Kv: LlamaKvCacheType.F16,
                Flash: LlamaFlashAttention.Auto,
                GpuLayers: 0);
            probeNotes = "GPU unavailable — planning CPU-only.";
        }
        else
        {
            (winner, probeNotes) = ProbeFullOffload(modelPath, candidates);
            if (winner is null)
            {
                // Model doesn't fit on GPU with any candidate ctx even at
                // the smallest tried. Fall back to partial-offload
                // heuristic, which plans layers based on weight-per-layer
                // math rather than empirical probing (probing each
                // candidate-layer-count combination would multiply probe
                // time). Users on under-VRAM'd hardware get a reasonable
                // starting point that they can dial down from.
                (winner, probeNotes) = PlanPartialOffload(
                    modelBytes, nLayers, freeVramBytes, targetCtx);
            }
        }

        uint planCtx   = winner.Ctx;
        var  kvType    = winner.Kv;
        var  flash     = winner.Flash;
        int  gpuLayers = winner.GpuLayers;

        return new AutoConfigureResult(
            Load: new ModelLoadSettings
            {
                ModelPath         = modelPath,
                ContextSize       = planCtx,
                GpuLayerCount     = gpuLayers,
                LogicalBatchSize  = 512,
                PhysicalBatchSize = 512,
                UseMmap           = true,
                UseMlock          = false,
                OffloadKQV        = true,
                FlashAttention    = flash,
                KvCacheTypeK      = kvType,
                KvCacheTypeV      = kvType,
            },
            Sampler: sampler,
            Generation: new GenerationSettings
            {
                // 1024 is cripplingly low for modern reasoning/thinking
                // models — a single <think> block routinely exceeds it.
                // 16 K gives plenty of runway and still terminates
                // runaway generations.
                MaxTokens = 16384,
            },
            Explanation: BuildExplanation(
                archName, humanName, devices, modelBytes, nLayers,
                gpuLayers, planCtx, kvType, samplerEntry, probeNotes));
    }

    private static List<ProbeCandidate> BuildCandidates(uint targetCtx)
    {
        var seen = new HashSet<uint>();
        var list = new List<ProbeCandidate>();
        foreach (var f in CtxCandidateFractions)
        {
            uint c = (uint)(targetCtx * f);
            c = Math.Max(MinContextSize, (c / 1024u) * 1024u);
            if (seen.Add(c))
                list.Add(new ProbeCandidate(c, LlamaKvCacheType.Q8_0));
        }
        if (seen.Add(MinContextSize))
            list.Add(new ProbeCandidate(MinContextSize, LlamaKvCacheType.Q8_0));
        return list;
    }

    /// <summary>
    /// Load the model once with <c>gpu_layers=-1</c>, then try each
    /// candidate context size in order. Returns the first one that
    /// successfully constructs a <see cref="LlamaContext"/>, or
    /// <c>(null, reason)</c> if the full-offload model load itself fails
    /// (weights don't fit on GPU).
    /// </summary>
    private static (ProbeResult?, string) ProbeFullOffload(
        string modelPath, List<ProbeCandidate> candidates)
    {
        LlamaModel? probe;
        try
        {
            probe = new LlamaModel(modelPath, new LlamaModelParameters
            {
                GpuLayerCount = -1,
                UseMmap = true,
                UseMlock = false,
            });
        }
        catch (LlamaException ex)
        {
            return (null, $"Full-offload load failed ({ex.Message}); falling back to partial offload.");
        }

        try
        {
            var attempted = new List<string>();
            foreach (var c in candidates)
            {
                var flash = c.Kv == LlamaKvCacheType.F16
                    ? LlamaFlashAttention.Auto
                    : LlamaFlashAttention.Enabled;
                try
                {
                    using var ctx = new LlamaContext(probe, new LlamaContextParameters
                    {
                        ContextSize        = c.Ctx,
                        LogicalBatchSize   = 512,
                        PhysicalBatchSize  = 512,
                        MaxSequenceCount   = 1,
                        OffloadKQV         = true,
                        FlashAttention     = flash,
                        KvCacheTypeK       = c.Kv,
                        KvCacheTypeV       = c.Kv,
                    });
                    var notes = attempted.Count == 0
                        ? $"Empirical probe: ctx={c.Ctx} {c.Kv} fit on first try."
                        : $"Empirical probe: stepped down through [{string.Join(", ", attempted)}], settled at ctx={c.Ctx} {c.Kv}.";
                    return (new ProbeResult(c.Ctx, c.Kv, flash, GpuLayers: -1), notes);
                }
                catch (LlamaException)
                {
                    attempted.Add($"{c.Ctx}×{c.Kv}");
                }
            }
            return (null,
                $"Every candidate OOM'd at full offload: [{string.Join(", ", attempted)}]. " +
                $"Falling back to partial offload.");
        }
        finally
        {
            probe.Dispose();
        }
    }

    /// <summary>
    /// Partial-offload fallback: the model's weights don't fit on GPU, so
    /// plan how many layers to offload using a simple bytes-per-layer
    /// heuristic plus a conservative KV estimate. Empirical probing of
    /// each possible layer count would multiply probe time by ~n_layers,
    /// which isn't worth it for a fallback path.
    /// </summary>
    private static (ProbeResult, string) PlanPartialOffload(
        long modelBytes, int nLayers, long freeVramBytes, uint targetCtx)
    {
        // Reserve 1 GiB of VRAM for activation scratch + allocator slop
        // on the partial-offload path — we're already making estimates,
        // so err on the safe side.
        const long partialReserveBytes = 1L << 30;
        long weightBudget = Math.Max(0, freeVramBytes - partialReserveBytes);
        long perLayerWeights = Math.Max(1, modelBytes / nLayers);
        int layers = (int)Math.Clamp(weightBudget / perLayerWeights, 0L, nLayers);

        return (
            new ProbeResult(
                Ctx: targetCtx,
                Kv: LlamaKvCacheType.Q8_0,
                Flash: LlamaFlashAttention.Enabled,
                GpuLayers: layers),
            $"Partial offload: planning {layers}/{nLayers} layers on GPU. " +
            $"Empirical probe declined to search layer counts; dial manually if this OOMs.");
    }

    private static SamplerSettings ApplyProfile(SamplerSettings baseline, SamplingProfileValues p) => baseline with
    {
        Temperature   = p.Temperature    ?? baseline.Temperature,
        TopP          = p.TopP           ?? baseline.TopP,
        TopK          = p.TopK           ?? baseline.TopK,
        MinP          = p.MinP           ?? baseline.MinP,
        PenaltyRepeat = p.PenaltyRepeat  ?? baseline.PenaltyRepeat,
    };

    private static string BuildExplanation(
        string? arch, string? name,
        IReadOnlyList<LlamaComputeDevice> devices,
        long modelBytes, int nLayers,
        int gpuLayers, uint ctx,
        LlamaKvCacheType kv,
        SamplingProfileEntry samplerEntry,
        string probeNotes)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append("Model: ").Append(name ?? "(unnamed)")
          .Append(" — arch=").Append(arch ?? "(unknown)")
          .Append(", size=").Append(FormatBytes(modelBytes))
          .Append(", layers=").Append(nLayers).AppendLine();

        long freeVram = 0;
        foreach (var d in devices)
            if (d.Type is LlamaComputeDeviceType.Gpu or LlamaComputeDeviceType.IntegratedGpu)
                freeVram += d.FreeBytes;
        sb.Append("Hardware: ").Append(devices.Count).Append(" device(s), free VRAM=")
          .Append(FormatBytes(freeVram)).AppendLine();

        sb.Append("Load: gpu_layers=")
          .Append(gpuLayers == -1 ? "all" : gpuLayers.ToString())
          .Append(", ctx=").Append(ctx)
          .Append(", kv=").Append(kv)
          .AppendLine();

        sb.Append(probeNotes).AppendLine();

        sb.Append("Sampler: ").Append(samplerEntry.Id);
        if (!string.IsNullOrEmpty(samplerEntry.Notes))
            sb.Append(" — ").Append(samplerEntry.Notes);
        return sb.ToString();
    }

    private static string FormatBytes(long b) => b switch
    {
        >= 1L << 30 => $"{b / (double)(1L << 30):F1} GiB",
        >= 1L << 20 => $"{b / (double)(1L << 20):F0} MiB",
        _           => $"{b} B",
    };

    private readonly record struct ProbeCandidate(uint Ctx, LlamaKvCacheType Kv);

    private sealed record ProbeResult(
        uint Ctx,
        LlamaKvCacheType Kv,
        LlamaFlashAttention Flash,
        int GpuLayers);
}

public sealed record AutoConfigureResult(
    ModelLoadSettings Load,
    SamplerSettings Sampler,
    GenerationSettings Generation,
    string Explanation);
