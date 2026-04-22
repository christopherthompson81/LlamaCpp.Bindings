using LlamaCpp.Bindings.Native;
using LlamaCpp.Bindings.Native.SafeHandles;

namespace LlamaCpp.Bindings;

/// <summary>
/// A per-session inference context. Owns a KV cache; one context serves one
/// conversation at a time (the Phase-1 shape — Phase 4 adds KV-cache reset and
/// sequence-removal for multi-turn chat).
/// </summary>
/// <remarks>
/// Must be disposed before its backing <see cref="LlamaModel"/>.
/// </remarks>
public sealed class LlamaContext : IDisposable
{
    private readonly SafeLlamaContextHandle _handle;
    private readonly LlamaModel _model;
    private bool _disposed;

    /// <summary>Actual context size the native library is using (may differ from requested).</summary>
    public int ContextSize { get; }

    /// <summary>Actual logical batch size.</summary>
    public int LogicalBatchSize { get; }

    /// <summary>Actual physical batch size.</summary>
    public int PhysicalBatchSize { get; }

    /// <summary>Max concurrent sequences.</summary>
    public int MaxSequenceCount { get; }

    public LlamaModel Model
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _model;
        }
    }

    public LlamaContext(LlamaModel model, LlamaContextParameters? parameters = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        LlamaBackend.EnsureInitialized();

        _model = model;
        var native = (parameters ?? LlamaContextParameters.Default()).ToNative();

        var raw = NativeMethods.llama_init_from_model(model.Handle.DangerousHandle, native);
        if (raw == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_init_from_model),
                "llama_init_from_model returned NULL. Common causes: requested n_ctx exceeds VRAM/RAM budget, " +
                "unsupported quant for this backend, or model+params incompatibility. Check the native log.");
        }

        _handle = SafeLlamaContextHandle.FromUnsafeHandle(raw);

        // Query back the actual values — llama.cpp may have adjusted them.
        ContextSize       = (int)NativeMethods.llama_n_ctx(raw);
        LogicalBatchSize  = (int)NativeMethods.llama_n_batch(raw);
        PhysicalBatchSize = (int)NativeMethods.llama_n_ubatch(raw);
        MaxSequenceCount  = (int)NativeMethods.llama_n_seq_max(raw);
    }

    internal SafeLlamaContextHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    // ----- KV cache management (Phase 4) -----
    //
    // The KV cache holds the attention state for every position we've already
    // decoded. Chat-style use leans on it hard: turn N + 1 should only decode
    // the newly-spoken tokens, with turn N's prefix still present in the cache.
    // Phase 4 exposes just the primitives the kickoff calls out; richer APIs
    // (sequence branching, RoPE position shifts) live behind the raw memory
    // getter for advanced callers that land later.

    private IntPtr _memoryHandleCache = IntPtr.Zero;
    private bool _memoryProbed;

    /// <summary>
    /// True when this context has an attached KV / recurrent memory.
    /// Encoder-only models (BGE, reranker, T5 encoder) have no memory —
    /// every call reads the input fresh. Calling <see cref="ClearKvCache"/>
    /// or any other memory-dependent method on such a context is a no-op
    /// rather than an error.
    /// </summary>
    public bool HasKvCache
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ProbeMemory();
            return _memoryHandleCache != IntPtr.Zero;
        }
    }

    /// <summary>
    /// Get the native memory handle, or throw if this context has none
    /// and the caller required memory. Internal: callers that can tolerate
    /// memory-less contexts should branch on <see cref="HasKvCache"/>.
    /// </summary>
    private IntPtr Memory()
    {
        ProbeMemory();
        if (_memoryHandleCache == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_get_memory),
                "This context has no attached KV cache (encoder-only model). " +
                "Check HasKvCache before calling memory-dependent methods.");
        }
        return _memoryHandleCache;
    }

    private void ProbeMemory()
    {
        if (_memoryProbed) return;
        _memoryHandleCache = NativeMethods.llama_get_memory(_handle.DangerousHandle);
        _memoryProbed = true;
    }

    /// <summary>
    /// Zero out the KV cache for every sequence. Equivalent to starting a
    /// fresh context without the cost of re-allocating GPU buffers. Use this
    /// to begin an unrelated conversation.
    /// </summary>
    /// <param name="alsoClearData">
    /// If true (the default) the data buffers are wiped alongside metadata.
    /// Set false only when you know you'll overwrite every slot before use —
    /// it saves a memset but leaves stale data readable to inspection tools.
    /// </param>
    public void ClearKvCache(bool alsoClearData = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!HasKvCache) return; // encoder-only model — nothing to clear
        NativeMethods.llama_memory_clear(Memory(), alsoClearData);
    }

    /// <summary>
    /// Remove all tokens in sequence <paramref name="sequenceId"/> with
    /// positions in <c>[fromPosition, toPosition)</c>.
    /// </summary>
    /// <param name="sequenceId">Sequence to prune. Negative matches any sequence.</param>
    /// <param name="fromPosition">Inclusive lower bound. Negative = <c>0</c>.</param>
    /// <param name="toPosition">Exclusive upper bound. Negative = end-of-sequence.</param>
    /// <returns>
    /// True on success. False if the range couldn't be removed cleanly (e.g. a
    /// partial block inside a quantised KV cache). Removing an entire sequence
    /// always succeeds.
    /// </returns>
    public bool RemoveSequenceRange(int sequenceId = 0, int fromPosition = 0, int toPosition = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeMethods.llama_memory_seq_rm(Memory(), sequenceId, fromPosition, toPosition);
    }

    /// <summary>
    /// Copy every token from <paramref name="sourceSequenceId"/> in
    /// <c>[fromPosition, toPosition)</c> to
    /// <paramref name="destinationSequenceId"/>. Useful for branching (e.g.
    /// speculative decoding) — turns one prefix into two that diverge from
    /// position <paramref name="toPosition"/>.
    /// </summary>
    public void CopySequence(
        int sourceSequenceId,
        int destinationSequenceId,
        int fromPosition = 0,
        int toPosition = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_memory_seq_cp(Memory(), sourceSequenceId, destinationSequenceId, fromPosition, toPosition);
    }

    /// <summary>
    /// Remove every token that doesn't belong to <paramref name="sequenceId"/>.
    /// Cheaper than clearing and re-decoding when you want to discard all
    /// branches except one.
    /// </summary>
    public void KeepOnlySequence(int sequenceId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_memory_seq_keep(Memory(), sequenceId);
    }

    /// <summary>
    /// Returns the lowest and highest positions present in the KV cache for
    /// <paramref name="sequenceId"/>, or <c>(null, null)</c> if the sequence
    /// is empty. Every position in the range is guaranteed present.
    /// </summary>
    public (int? Minimum, int? Maximum) SequencePositionRange(int sequenceId = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var mem = Memory();
        var min = NativeMethods.llama_memory_seq_pos_min(mem, sequenceId);
        var max = NativeMethods.llama_memory_seq_pos_max(mem, sequenceId);
        return (min < 0 ? null : min, max < 0 ? null : max);
    }

    /// <summary>
    /// Whether the underlying memory backend supports RoPE position shifts —
    /// used by tricks like sliding-window truncation. Most backends return
    /// true; quantised KV caches can return false.
    /// </summary>
    public bool SupportsPositionShift()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeMethods.llama_memory_can_shift(Memory());
    }

    /// <summary>
    /// Shift every token position in <paramref name="sequenceId"/> in range
    /// <c>[fromPosition, toPosition)</c> by <paramref name="delta"/>. Delta
    /// may be negative to shift positions earlier.
    /// </summary>
    /// <remarks>
    /// The canonical use is sliding-window truncation: drop the oldest K
    /// tokens with <see cref="RemoveSequenceRange"/>, then call this with
    /// <c>(fromPosition = K, toPosition = -1, delta = -K)</c> to renumber
    /// the surviving tokens back to starting at 0. Requires the memory
    /// backend to support shifts — check <see cref="SupportsPositionShift"/>
    /// first on exotic quantised-KV configurations.
    /// </remarks>
    public void ShiftSequencePositions(
        int sequenceId, int fromPosition, int toPosition, int delta)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureBackendSupportsShift(nameof(ShiftSequencePositions));
        NativeMethods.llama_memory_seq_add(Memory(), sequenceId, fromPosition, toPosition, delta);
    }

    /// <summary>
    /// Divide every token position in range <c>[fromPosition, toPosition)</c>
    /// of <paramref name="sequenceId"/> by <paramref name="divisor"/>. Rarely
    /// useful — enables aggressive schemes that re-map positions onto a
    /// coarser grid. <paramref name="divisor"/> must be ≥ 1.
    /// </summary>
    public void DivideSequencePositions(
        int sequenceId, int fromPosition, int toPosition, int divisor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (divisor < 1) throw new ArgumentOutOfRangeException(nameof(divisor), "must be ≥ 1");
        EnsureBackendSupportsShift(nameof(DivideSequencePositions));
        NativeMethods.llama_memory_seq_div(Memory(), sequenceId, fromPosition, toPosition, divisor);
    }

    /// <summary>
    /// Both <see cref="ShiftSequencePositions"/> and
    /// <see cref="DivideSequencePositions"/> assert at the native level on
    /// MRope/IMRope models and other configurations where
    /// <c>n_pos_per_embd() &gt; 1</c>. The native assert is a process-level
    /// abort — we refuse to make the call if we know it's going to fail.
    /// </summary>
    private void EnsureBackendSupportsShift(string callerName)
    {
        if (!NativeMethods.llama_memory_can_shift(Memory()))
        {
            throw new NotSupportedException(
                $"{callerName} is not supported by this model's KV cache configuration " +
                $"(llama_memory_can_shift returned false). Typical causes: multi-dimensional " +
                $"position encodings (MRope/IMRope, used by Qwen3, some vision models), or " +
                $"quantised KV caches that lack a shift implementation. Use " +
                $"RemoveSequenceRange + re-decode to achieve truncation instead.");
        }
    }

    /// <summary>
    /// Ask llama.cpp to log a per-device memory-usage breakdown via its log
    /// sink. Diagnostic tool — useful when diagnosing VRAM-overallocation
    /// failures.
    /// </summary>
    public void LogMemoryBreakdown()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_memory_breakdown_print(_handle.DangerousHandle);
    }

    // ----- Runtime settings + mid-flight getters (Tier 1 expansion) -----

    /// <summary>
    /// Current decode-thread count. Returns the value set at construction
    /// (<see cref="LlamaContextParameters.ThreadCount"/>) unless later
    /// overridden via <see cref="SetThreadCounts"/>.
    /// </summary>
    public int ThreadCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.llama_n_threads(_handle.DangerousHandle);
        }
    }

    /// <summary>Thread count used for prompt/batch processing.</summary>
    public int BatchThreadCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.llama_n_threads_batch(_handle.DangerousHandle);
        }
    }

    /// <summary>Live update to the context's thread counts. Takes effect on the next decode.</summary>
    public void SetThreadCounts(int generationThreads, int batchThreads)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_set_n_threads(_handle.DangerousHandle, generationThreads, batchThreads);
    }

    /// <summary>Pooling strategy for embedding extraction.</summary>
    public LlamaPoolingType PoolingType
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return (LlamaPoolingType)NativeMethods.llama_pooling_type(_handle.DangerousHandle);
        }
    }

    /// <summary>
    /// Per-sequence context length. For single-sequence contexts this is the
    /// same as <see cref="ContextSize"/>; for multi-sequence contexts this
    /// tells you how much context each sequence gets.
    /// </summary>
    public int SequenceContextSize
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return (int)NativeMethods.llama_n_ctx_seq(_handle.DangerousHandle);
        }
    }

    /// <summary>
    /// Toggle the embeddings-output mode on this live context. When enabled,
    /// <c>llama_decode</c> makes embeddings available (via functions we haven't
    /// bound yet — see issue "Embeddings support" in the Tier-2 tracker).
    /// </summary>
    public void SetEmbeddingsMode(bool enabled)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_set_embeddings(_handle.DangerousHandle, enabled);
    }

    /// <summary>
    /// Toggle causal attention. Default is <c>true</c> (standard left-to-right
    /// generation). Setting <c>false</c> is only meaningful for non-generative
    /// workflows (bidirectional encoder-style models).
    /// </summary>
    public void SetCausalAttention(bool enabled)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_set_causal_attn(_handle.DangerousHandle, enabled);
    }

    /// <summary>
    /// Flag the context as "warming up" — llama.cpp uses this to skip certain
    /// optimisations that expect a real workload. Useful when doing a quick
    /// dummy decode to pre-allocate buffers.
    /// </summary>
    public void SetWarmup(bool enabled)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_set_warmup(_handle.DangerousHandle, enabled);
    }

    /// <summary>
    /// Block until all pending async native work on this context has finished.
    /// Useful right before calling <see cref="GetPerformance"/> or other code
    /// that needs a stable snapshot.
    /// </summary>
    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_synchronize(_handle.DangerousHandle);
    }

    // ----- Embeddings (Tier 2 expansion) -----

    /// <summary>
    /// Per-token embedding for the <paramref name="tokenIndex"/>-th output of
    /// the last decode, copied into a managed array. Returns null if the
    /// index is invalid or the context wasn't configured for embeddings.
    /// </summary>
    /// <remarks>
    /// Negative indices count from the end (<c>-1</c> = last output, just like
    /// slicing in most languages).
    /// </remarks>
    public unsafe float[]? GetTokenEmbedding(int tokenIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ptr = NativeMethods.llama_get_embeddings_ith(_handle.DangerousHandle, tokenIndex);
        if (ptr == null) return null;
        int dim = _model.EmbeddingSize;
        var copy = new float[dim];
        new ReadOnlySpan<float>(ptr, dim).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Pooled embedding for <paramref name="sequenceId"/>, copied into a
    /// managed array. Returns null when the context's pooling mode is
    /// <see cref="LlamaPoolingType.None"/> or the sequence has no output.
    /// </summary>
    /// <remarks>
    /// For classifier / reranker models (pooling = Rank) the returned array
    /// has length <see cref="LlamaModel.ClassifierOutputCount"/> instead of
    /// <see cref="LlamaModel.EmbeddingSize"/>.
    /// </remarks>
    public unsafe float[]? GetSequenceEmbedding(int sequenceId = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ptr = NativeMethods.llama_get_embeddings_seq(_handle.DangerousHandle, sequenceId);
        if (ptr == null) return null;
        int dim = PoolingType == LlamaPoolingType.Rank
            ? Math.Max(1, _model.ClassifierOutputCount)
            : _model.EmbeddingSize;
        var copy = new float[dim];
        new ReadOnlySpan<float>(ptr, dim).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Run the model's encoder tower over a batch of tokens. Only meaningful
    /// for encoder-decoder models (<see cref="LlamaModel.HasEncoder"/> true —
    /// T5 family). For decoder-only models use <see cref="EncodeForEmbedding"/>
    /// or call into the decode path with embeddings mode enabled.
    /// </summary>
    /// <returns>Native status code. 0 = success, negative = error.</returns>
    public unsafe int RunEncoder(ReadOnlySpan<int> tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return 0;

        var array = tokens.ToArray();
        fixed (int* ptr = array)
        {
            var batch = NativeMethods.llama_batch_get_one(ptr, array.Length);
            return NativeMethods.llama_encode(_handle.DangerousHandle, batch);
        }
    }

    /// <summary>
    /// Convenience: tokenize <paramref name="text"/>, enable embeddings mode,
    /// decode it once (or run the encoder for encoder-decoder models), and
    /// return the pooled sequence embedding. Useful for one-shot
    /// retrieval/RAG workflows where you only need a single vector per input.
    /// </summary>
    /// <param name="text">Input text. Empty returns an empty array.</param>
    /// <param name="addSpecial">Forward to Vocab.Tokenize.</param>
    /// <param name="parseSpecial">Forward to Vocab.Tokenize.</param>
    /// <returns>
    /// The pooled embedding vector, or an empty array if the context doesn't
    /// produce a sequence-pooled result (pooling = None).
    /// </returns>
    public unsafe float[] EncodeForEmbedding(
        string text,
        bool addSpecial = true,
        bool parseSpecial = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0) return Array.Empty<float>();

        SetEmbeddingsMode(true);
        ClearKvCache();

        var tokens = _model.Vocab.Tokenize(text, addSpecial, parseSpecial);

        int rc;
        fixed (int* tokPtr = tokens)
        {
            var batch = NativeMethods.llama_batch_get_one(tokPtr, tokens.Length);
            // Encoder-decoder models: run encoder. Decoder-only: run decode
            // with embeddings mode on — the hidden states land in the
            // embedding buffer rather than logits.
            rc = _model.HasEncoder
                ? NativeMethods.llama_encode(_handle.DangerousHandle, batch)
                : NativeMethods.llama_decode(_handle.DangerousHandle, batch);
        }
        if (rc != 0)
        {
            throw new LlamaException(
                _model.HasEncoder ? nameof(NativeMethods.llama_encode) : nameof(NativeMethods.llama_decode),
                rc,
                "Embedding pass failed. Check the native log and the context's embeddings/pooling configuration.");
        }

        return GetSequenceEmbedding(0)
            ?? Array.Empty<float>();
    }

    // ----- Performance (Tier 1 expansion) -----

    /// <summary>
    /// Snapshot the context's timing counters. Cheap to call (~tens of ns);
    /// safe to sample frequently.
    /// </summary>
    public LlamaContextPerformance GetPerformance()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var d = NativeMethods.llama_perf_context(_handle.DangerousHandle);
        return new LlamaContextPerformance(
            StartMilliseconds:      d.t_start_ms,
            LoadMilliseconds:       d.t_load_ms,
            PromptEvalMilliseconds: d.t_p_eval_ms,
            TokenEvalMilliseconds:  d.t_eval_ms,
            PromptTokenCount:       d.n_p_eval,
            GeneratedTokenCount:    d.n_eval,
            GraphReuseCount:        d.n_reused);
    }

    /// <summary>
    /// Reset all performance counters to zero. The next call to
    /// <see cref="GetPerformance"/> measures from this point forward.
    /// </summary>
    public void ResetPerformance()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_perf_context_reset(_handle.DangerousHandle);
    }

    /// <summary>
    /// Ask llama.cpp to log a human-readable performance report via its log
    /// sink. Primarily a diagnostic shortcut; prefer <see cref="GetPerformance"/>
    /// for programmatic use.
    /// </summary>
    public void LogPerformanceReport()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_perf_context_print(_handle.DangerousHandle);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _memoryHandleCache = IntPtr.Zero;
        _handle.Dispose();
    }
}
