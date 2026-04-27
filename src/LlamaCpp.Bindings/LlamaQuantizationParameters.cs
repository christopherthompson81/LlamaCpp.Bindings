using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Target file type for <see cref="LlamaQuantizer"/>. Values match the
/// <c>LLAMA_FTYPE_*</c> integer constants so they can be cast directly into
/// the native struct. Numbering has gaps where llama.cpp removed types over
/// time — preserve the exact integer values when adding new entries.
/// </summary>
public enum LlamaFileType
{
    /// <summary>All tensors in F32. Largest, exactly preserves the source.</summary>
    AllF32       = 0,
    /// <summary>F16 weights, F32 1-D tensors. Lossless for most pretrained checkpoints.</summary>
    MostlyF16    = 1,
    /// <summary>bfloat16 weights, F32 1-D tensors.</summary>
    BF16         = 32,
    Q4_0         = 2,
    Q4_1         = 3,
    Q5_0         = 8,
    Q5_1         = 9,
    Q8_0         = 7,
    Q1_0         = 40,
    Q2_K         = 10,
    Q2_K_S       = 21,
    Q3_K_S       = 11,
    Q3_K_M       = 12,
    Q3_K_L       = 13,
    Q4_K_S       = 14,
    /// <summary>The most popular general-purpose quant — ~4.5 bits/weight, near-FP16 quality.</summary>
    Q4_K_M       = 15,
    Q5_K_S       = 16,
    Q5_K_M       = 17,
    Q6_K         = 18,
    IQ1_S        = 24,
    IQ1_M        = 31,
    IQ2_XXS      = 19,
    IQ2_XS       = 20,
    IQ2_S        = 28,
    IQ2_M        = 29,
    IQ3_XXS      = 23,
    IQ3_XS       = 22,
    IQ3_S        = 26,
    IQ3_M        = 27,
    IQ4_NL       = 25,
    IQ4_XS       = 30,
    TQ1_0        = 36,
    TQ2_0        = 37,
    /// <summary>4-bit MXFP4 for MoE expert weights (used by GPT-OSS-style mixtures).</summary>
    Mxfp4Moe     = 38,
    Nvfp4        = 39,
    /// <summary>Sentinel: ftype unknown / read from the source file.</summary>
    Guessed      = 1024,
}

/// <summary>
/// Tensor element type for <see cref="LlamaQuantizationParameters.OutputTensorType"/>
/// and <see cref="LlamaQuantizationParameters.TokenEmbeddingType"/> overrides.
/// Mirrors the subset of <c>ggml_type</c> values accepted by
/// <c>llama_model_quantize</c>. Integer values match the underlying
/// <c>ggml_type</c> constants for direct casts.
/// </summary>
public enum LlamaTensorType
{
    F32     = 0,
    F16     = 1,
    BF16    = 30,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    IQ1_S   = 19,
    IQ1_M   = 29,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ2_S   = 22,
    IQ3_XXS = 18,
    IQ3_S   = 21,
    IQ4_NL  = 20,
    IQ4_XS  = 23,
    TQ1_0   = 34,
    TQ2_0   = 35,
    Mxfp4   = 39,
    Nvfp4   = 40,
}

/// <summary>
/// Knobs controlling a single <see cref="LlamaQuantizer.QuantizeAsync"/> call.
/// Mirrors the public subset of <c>llama_model_quantize_params</c>; defaults
/// match <c>llama_model_quantize_default_params()</c>.
/// </summary>
public sealed class LlamaQuantizationParameters
{
    /// <summary>Target ftype the output GGUF will be tagged with.</summary>
    public LlamaFileType FileType { get; set; } = LlamaFileType.Q5_1;

    /// <summary>
    /// Worker threads. <c>0</c> (default) uses
    /// <c>std::thread::hardware_concurrency()</c> on the native side.
    /// </summary>
    public int ThreadCount { get; set; } = 0;

    /// <summary>
    /// Override the type used for the <c>output.weight</c> tensor.
    /// <c>null</c> (default) leaves it to the per-ftype default — usually
    /// the same as <see cref="FileType"/>'s base type.
    /// </summary>
    public LlamaTensorType? OutputTensorType { get; set; }

    /// <summary>
    /// Override the type used for the token embedding tensor. <c>null</c>
    /// (default) preserves the per-ftype default.
    /// </summary>
    public LlamaTensorType? TokenEmbeddingType { get; set; }

    /// <summary>Allow re-quantizing an already-quantized source.</summary>
    public bool AllowRequantize { get; set; } = false;

    /// <summary>
    /// Quantize the <c>output.weight</c> tensor. Default <c>true</c>; setting
    /// false keeps it at its source type, which costs a few MB but slightly
    /// improves perplexity at small sizes.
    /// </summary>
    public bool QuantizeOutputTensor { get; set; } = true;

    /// <summary>
    /// Skip quantization entirely and just copy the tensors. Useful for
    /// stripping metadata or splitting/un-splitting a GGUF without changing
    /// quant. <see cref="FileType"/>, <see cref="AllowRequantize"/> and
    /// <see cref="QuantizeOutputTensor"/> are ignored when set.
    /// </summary>
    public bool OnlyCopy { get; set; } = false;

    /// <summary>
    /// Use the chosen ftype for every tensor — disable the per-tensor
    /// "smart" overrides llama.cpp normally applies (e.g. keeping
    /// <c>output.weight</c> at a higher precision than the body).
    /// </summary>
    public bool Pure { get; set; } = false;

    /// <summary>
    /// Preserve the source's shard count. If the input is split across
    /// multiple files, the output will be split with the same boundaries.
    /// </summary>
    public bool KeepSplit { get; set; } = false;

    /// <summary>
    /// Calculate the final size and tensor type breakdown without writing
    /// any output. Useful for size estimates in the GUI.
    /// </summary>
    public bool DryRun { get; set; } = false;

    /// <summary>
    /// Per-tensor type overrides applied by name pattern. Each entry
    /// pins matching tensors to a specific element type, regardless
    /// of <see cref="FileType"/>'s default for that role. Patterns
    /// are <strong>ECMAScript regular expressions</strong> matched
    /// against the GGUF tensor name via substring search; the first
    /// matching entry wins.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <strong>Important:</strong> upstream gates the override path
    /// behind <c>!Pure</c> — if <see cref="Pure"/> is <c>true</c>, the
    /// override list is ignored. Set <see cref="Pure"/> to <c>false</c>
    /// (the default) when using overrides.
    /// </para>
    /// <para>
    /// Patterns are regex, not globs. A pattern like
    /// <c>"blk.0.attn_q.weight"</c> treats every <c>.</c> as
    /// "any character" — it would also match <c>"blkX0Yattn_q_weight"</c>.
    /// Escape literal dots and anchor when intent is exact:
    /// <c>"^blk\.0\.attn_q\.weight$"</c>. Per-layer-group patterns
    /// like <c>"blk\.\d+\.attn_q\.weight"</c> are the natural form.
    /// </para>
    /// <para>
    /// This is the "recipe" channel for the Adaptive Quantization
    /// pipeline: a per-tensor sensitivity sweep computes a
    /// {tensor → ftype} map, and the recipe-applier passes it through
    /// here. It's also useful standalone — pin
    /// <c>output.weight</c> at <c>Q8_0</c> to keep the LM head higher
    /// precision than the body, or pin a specific layer's attention
    /// QKV at a different type than the rest.
    /// </para>
    /// </remarks>
    public IReadOnlyList<KeyValuePair<string, LlamaTensorType>>? TensorTypeOverrides { get; set; }

    /// <summary>
    /// Optional path to an importance-matrix GGUF (e.g. produced by
    /// <see cref="LlamaImatrix.ComputeAsync"/>). When set, the per-tensor
    /// column-importance values are forwarded to <c>llama_model_quantize</c>
    /// so the per-block scale optimization is imatrix-aware. Without this,
    /// even ftypes that nominally need imatrix (Q2_K, IQ-quants) run
    /// unweighted and the quantizer warns. Pairs cleanly with
    /// <see cref="TensorTypeOverrides"/> from a recipe built against the
    /// same imatrix.
    /// </summary>
    public string? ImatrixPath { get; set; }

    /// <summary>
    /// Snapshot of the native defaults as a managed parameters object.
    /// </summary>
    public static LlamaQuantizationParameters Default()
    {
        LlamaBackend.EnsureInitialized();
        var native = NativeMethods.llama_model_quantize_default_params();
        return new LlamaQuantizationParameters
        {
            FileType             = (LlamaFileType)native.ftype,
            ThreadCount          = native.nthread,
            // The native sentinel for "unset" is GGML_TYPE_COUNT, which is
            // outside the LlamaTensorType range; we surface that as null.
            OutputTensorType     = TryMapTensorType(native.output_tensor_type),
            TokenEmbeddingType   = TryMapTensorType(native.token_embedding_type),
            AllowRequantize      = native.allow_requantize,
            QuantizeOutputTensor = native.quantize_output_tensor,
            OnlyCopy             = native.only_copy,
            Pure                 = native.pure,
            KeepSplit            = native.keep_split,
            DryRun               = native.dry_run,
        };
    }

    internal llama_model_quantize_params ToNative()
    {
        var native = NativeMethods.llama_model_quantize_default_params();
        native.ftype                  = (llama_ftype)FileType;
        native.nthread                = ThreadCount;
        if (OutputTensorType is { } ot)
            native.output_tensor_type = (ggml_type)ot;
        if (TokenEmbeddingType is { } et)
            native.token_embedding_type = (ggml_type)et;
        native.allow_requantize       = AllowRequantize;
        native.quantize_output_tensor = QuantizeOutputTensor;
        native.only_copy              = OnlyCopy;
        native.pure                   = Pure;
        native.keep_split             = KeepSplit;
        native.dry_run                = DryRun;
        // imatrix / kv_overrides / tt_overrides / prune_layers all stay at
        // their default null pointers — V1 doesn't surface those yet.
        return native;
    }

    private static LlamaTensorType? TryMapTensorType(ggml_type t)
    {
        // GGML_TYPE_COUNT is the "unset" sentinel returned by
        // llama_model_quantize_default_params for the override fields. It
        // isn't part of our public LlamaTensorType — translate to null.
        return Enum.IsDefined(typeof(LlamaTensorType), (int)t)
            ? (LlamaTensorType)(int)t
            : null;
    }
}
