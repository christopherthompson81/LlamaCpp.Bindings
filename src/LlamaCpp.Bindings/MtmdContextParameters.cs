using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// User-facing overrides for <see cref="MtmdContext"/> construction.
/// Mirrors the knobs on native <c>mtmd_context_params</c>; any field left
/// unset keeps the default returned by <c>mtmd_context_params_default</c>.
/// </summary>
/// <remarks>
/// The media-marker strings (<c>ImageMarker</c>, <c>MediaMarker</c>) are kept
/// as-is from the native defaults — we don't expose knobs for them because
/// every call site reads <see cref="MtmdContext.DefaultMediaMarker"/> and
/// splices it into the prompt itself.
/// </remarks>
public sealed class MtmdContextParameters
{
    /// <summary>Offload the vision encoder to GPU. Default: true.</summary>
    public bool? UseGpu { get; set; }

    /// <summary>Log per-image timing to the native log stream. Default: false.</summary>
    public bool? PrintTimings { get; set; }

    /// <summary>Threads for CPU-side encode. Default: 4.</summary>
    public int? ThreadCount { get; set; }

    /// <summary>Flash-attention mode. Default: auto (let the runtime decide).</summary>
    public LlamaFlashAttentionType? FlashAttentionType { get; set; }

    /// <summary>
    /// Run a warmup encode pass on construction. Default: true. Disabling can
    /// halve cold-start time for test fixtures where throughput isn't needed.
    /// </summary>
    public bool? Warmup { get; set; }

    /// <summary>
    /// Minimum image tokens for dynamic-resolution models. Default: read from
    /// model metadata (left as whatever the native default produces).
    /// </summary>
    public int? ImageMinTokens { get; set; }

    /// <summary>
    /// Maximum image tokens for dynamic-resolution models. Default: read from
    /// model metadata.
    /// </summary>
    public int? ImageMaxTokens { get; set; }

    internal mtmd_context_params ToNative()
    {
        var native = NativeMethods.mtmd_context_params_default();
        if (UseGpu.HasValue)             native.use_gpu = UseGpu.Value;
        if (PrintTimings.HasValue)       native.print_timings = PrintTimings.Value;
        if (ThreadCount.HasValue)        native.n_threads = ThreadCount.Value;
        if (FlashAttentionType.HasValue) native.flash_attn_type = (llama_flash_attn_type)FlashAttentionType.Value;
        if (Warmup.HasValue)             native.warmup = Warmup.Value;
        if (ImageMinTokens.HasValue)     native.image_min_tokens = ImageMinTokens.Value;
        if (ImageMaxTokens.HasValue)     native.image_max_tokens = ImageMaxTokens.Value;
        return native;
    }
}

/// <summary>Mirror of <c>llama_flash_attn_type</c> for public consumers.</summary>
public enum LlamaFlashAttentionType
{
    Auto     = -1,
    Disabled = 0,
    Enabled  = 1,
}
