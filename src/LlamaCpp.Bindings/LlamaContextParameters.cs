using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Managed-side mirror of the commonly-used subset of <c>llama_context_params</c>.
/// Defaults match <c>llama_context_default_params()</c> unless noted.
/// </summary>
public sealed class LlamaContextParameters
{
    /// <summary>
    /// Context length in tokens. <c>0</c> = use the model's training context (from GGUF metadata).
    /// The llama.cpp default is 512, but setting 0 is the usual right answer for chat — we inherit
    /// the model's full training context unless the host explicitly picks something smaller.
    /// </summary>
    public uint ContextSize { get; set; } = 0;

    /// <summary>Logical maximum batch size passed to <c>llama_decode</c>.</summary>
    public uint LogicalBatchSize { get; set; } = 512;

    /// <summary>Physical maximum batch size (the actual forward-pass chunk).</summary>
    public uint PhysicalBatchSize { get; set; } = 512;

    /// <summary>Max number of distinct sequences (for recurrent models / multi-session).</summary>
    public uint MaxSequenceCount { get; set; } = 1;

    /// <summary><c>0</c> = llama.cpp default; negative values are invalid. Set to your core count for CPU-bound workloads.</summary>
    public int ThreadCount { get; set; } = -1;

    /// <summary>Threads for batch (prompt-processing) passes. -1 = same as <see cref="ThreadCount"/>.</summary>
    public int BatchThreadCount { get; set; } = -1;

    /// <summary>Flash Attention. Auto lets llama.cpp pick per model; Enabled forces it on.</summary>
    public LlamaFlashAttention FlashAttention { get; set; } = LlamaFlashAttention.Auto;

    /// <summary>Extract embeddings alongside logits.</summary>
    public bool Embeddings { get; set; } = false;

    /// <summary>Offload KQV ops (including the KV cache) to GPU.</summary>
    public bool OffloadKQV { get; set; } = true;

    /// <summary>
    /// Use a full-size SWA (sliding-window attention) cache rather than the
    /// default compact one. llama.cpp's default (<c>false</c>) is more memory
    /// efficient but makes <see cref="LlamaContext.RemoveSequenceRange"/>
    /// return false at block boundaries and can leave <c>min &gt; 0</c> once
    /// the window fills. For single-user chat with multi-turn editing, full
    /// SWA is the right tradeoff — we default to <c>true</c>. Set <c>false</c>
    /// for memory-constrained workloads where you don't edit the KV cache.
    /// </summary>
    public bool UseFullSwaCache { get; set; } = true;

    /// <summary>
    /// Record per-context performance counters (prompt/token eval ms, token
    /// counts). llama.cpp defaults this to OFF via an inverted <c>no_perf</c>
    /// field; we flip it to ON because <see cref="LlamaContext.GetPerformance"/>
    /// is cheap and most callers want the numbers available. Set <c>false</c>
    /// if you care about the sub-microsecond timing overhead.
    /// </summary>
    public bool MeasurePerformance { get; set; } = true;

    /// <summary>
    /// Element type for the K component of the KV cache. Default <c>F16</c>
    /// matches <c>llama_context_default_params()</c>. Quantized options
    /// (Q8_0 / Q5_0 / Q5_1 / Q4_0 / Q4_1 / IQ4_NL) roughly halve or quarter
    /// the cache footprint but require Flash Attention on the compute path —
    /// see <see cref="FlashAttention"/>. The llama.cpp header flags these
    /// fields EXPERIMENTAL; breakage across version bumps is possible.
    /// </summary>
    public LlamaKvCacheType KvCacheTypeK { get; set; } = LlamaKvCacheType.F16;

    /// <summary>Element type for the V component of the KV cache. See
    /// <see cref="KvCacheTypeK"/> for caveats.</summary>
    public LlamaKvCacheType KvCacheTypeV { get; set; } = LlamaKvCacheType.F16;

    internal llama_context_params ToNative()
    {
        var native = NativeMethods.llama_context_default_params();
        native.n_ctx = ContextSize;
        native.n_batch = LogicalBatchSize;
        native.n_ubatch = PhysicalBatchSize;
        native.n_seq_max = MaxSequenceCount;
        if (ThreadCount >= 0) native.n_threads = ThreadCount;
        if (BatchThreadCount >= 0) native.n_threads_batch = BatchThreadCount;
        native.flash_attn_type = (llama_flash_attn_type)FlashAttention;
        native.embeddings = Embeddings;
        native.offload_kqv = OffloadKQV;
        native.swa_full = UseFullSwaCache;
        native.no_perf = !MeasurePerformance;
        native.type_k = (ggml_type)(int)KvCacheTypeK;
        native.type_v = (ggml_type)(int)KvCacheTypeV;
        return native;
    }

    public static LlamaContextParameters Default()
    {
        LlamaBackend.EnsureInitialized();
        var n = NativeMethods.llama_context_default_params();
        return new LlamaContextParameters
        {
            ContextSize       = n.n_ctx,
            LogicalBatchSize  = n.n_batch,
            PhysicalBatchSize = n.n_ubatch,
            MaxSequenceCount  = n.n_seq_max,
            ThreadCount       = n.n_threads,
            BatchThreadCount  = n.n_threads_batch,
            FlashAttention    = (LlamaFlashAttention)n.flash_attn_type,
            Embeddings        = n.embeddings,
            OffloadKQV        = n.offload_kqv,
        };
    }
}

public enum LlamaFlashAttention
{
    Auto     = -1,
    Disabled = 0,
    Enabled  = 1,
}

/// <summary>
/// Subset of <c>ggml_type</c> values that llama.cpp accepts for the KV cache
/// K and V element types (mirrors the allow-list in llama-server's
/// <c>--cache-type-k/v</c> help output). Values match the underlying
/// <c>ggml_type</c> integer constants so we can cast directly.
/// </summary>
public enum LlamaKvCacheType
{
    /// <summary>32-bit float (largest, highest fidelity).</summary>
    F32    = 0,
    /// <summary>16-bit float — llama.cpp default. 2 bytes per element.</summary>
    F16    = 1,
    /// <summary>bfloat16 — 2 bytes per element, wider exponent than F16.</summary>
    BF16   = 30,
    /// <summary>8-bit quantization. ~1 byte per element (~50% smaller than F16). Needs Flash Attention.</summary>
    Q8_0   = 8,
    /// <summary>5-bit quantization. ~0.6 bytes per element. Needs Flash Attention.</summary>
    Q5_0   = 6,
    /// <summary>5-bit quantization, alternate form. ~0.6 bytes per element. Needs Flash Attention.</summary>
    Q5_1   = 7,
    /// <summary>4-bit quantization. ~0.5 bytes per element (~75% smaller than F16). Needs Flash Attention.</summary>
    Q4_0   = 2,
    /// <summary>4-bit quantization, alternate form. ~0.5 bytes per element. Needs Flash Attention.</summary>
    Q4_1   = 3,
    /// <summary>Importance-weighted 4-bit non-linear quantization. ~0.5 bytes per element. Needs Flash Attention.</summary>
    IQ4_NL = 20,
}

/// <summary>
/// How embedding outputs are pooled across a sequence. Only relevant when
/// the context is configured for embeddings (<see cref="LlamaContextParameters.Embeddings"/>
/// or <see cref="LlamaContext.SetEmbeddingsMode"/>).
/// </summary>
public enum LlamaPoolingType
{
    Unspecified = -1,
    /// <summary>No pooling — per-token embeddings are returned.</summary>
    None = 0,
    Mean = 1,
    /// <summary>Use the [CLS] token's embedding.</summary>
    Cls  = 2,
    /// <summary>Use the last token's embedding.</summary>
    Last = 3,
    /// <summary>Ranking head — used by reranker models.</summary>
    Rank = 4,
}
