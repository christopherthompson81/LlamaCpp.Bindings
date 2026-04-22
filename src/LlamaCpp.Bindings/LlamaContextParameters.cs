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
