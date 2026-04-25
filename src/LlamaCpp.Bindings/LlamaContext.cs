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

    // ----- State / session snapshots (Tier 2 expansion) -----
    //
    // Snapshot bytes capture the full context state (KV cache, logits buffer,
    // RNG state). They are tied to the exact pinned llama.cpp version: a
    // snapshot taken on version X is not guaranteed to load on version Y.
    // If you persist snapshots to disk, treat the pinned llama.cpp version
    // as part of the schema and invalidate old snapshots when you bump it.
    //
    // File variants stream directly through llama.cpp's native I/O — preferred
    // for large contexts where a round-trip through a managed byte[] would
    // allocate hundreds of megabytes (KV cache size scales with ctx × layers
    // × heads × kv_bits).

    /// <summary>Size in bytes required to hold a full-context snapshot.</summary>
    public long GetStateSize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return (long)NativeMethods.llama_state_get_size(_handle.DangerousHandle);
    }

    /// <summary>
    /// Snapshot the entire context state into a freshly-allocated array.
    /// Convenient for small contexts; for production use with large models
    /// prefer <see cref="SaveStateToFile"/> or the <see cref="Span{Byte}"/>
    /// overload with a pooled buffer.
    /// </summary>
    public byte[] SaveState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var size = NativeMethods.llama_state_get_size(_handle.DangerousHandle);
        var buffer = new byte[size];
        var written = SaveState(buffer);
        if (written != buffer.Length)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_state_get_data),
                $"Expected {buffer.Length} bytes, got {written}.");
        }
        return buffer;
    }

    /// <summary>
    /// Snapshot the context state into <paramref name="destination"/>. The
    /// buffer must be at least <see cref="GetStateSize"/> bytes; callers
    /// typically rent it from <c>ArrayPool&lt;byte&gt;.Shared</c>.
    /// </summary>
    /// <returns>Number of bytes actually written.</returns>
    public unsafe int SaveState(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var required = NativeMethods.llama_state_get_size(_handle.DangerousHandle);
        if ((nuint)destination.Length < required)
        {
            throw new ArgumentException(
                $"Destination buffer too small: need {required} bytes, have {destination.Length}.",
                nameof(destination));
        }
        fixed (byte* ptr = destination)
        {
            var written = NativeMethods.llama_state_get_data(
                _handle.DangerousHandle, ptr, (nuint)destination.Length);
            return checked((int)written);
        }
    }

    /// <summary>
    /// Restore context state from <paramref name="source"/>. The bytes must
    /// have been produced by <see cref="SaveState()"/> on a context built
    /// against the same pinned llama.cpp version.
    /// </summary>
    /// <returns>Number of bytes consumed.</returns>
    public unsafe int RestoreState(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.IsEmpty)
        {
            throw new ArgumentException("Snapshot is empty.", nameof(source));
        }
        fixed (byte* ptr = source)
        {
            var read = NativeMethods.llama_state_set_data(
                _handle.DangerousHandle, ptr, (nuint)source.Length);
            if (read == 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_state_set_data),
                    "Failed to restore state. The snapshot may be corrupt or from an " +
                    "incompatible llama.cpp version.");
            }
            return checked((int)read);
        }
    }

    /// <summary>
    /// Save the context state plus an optional associated token array to
    /// <paramref name="path"/>. The tokens are the prompt that produced the
    /// current KV state; storing them alongside lets a later load verify the
    /// cache is consistent with a prompt before reusing it.
    /// </summary>
    public unsafe void SaveStateToFile(string path, ReadOnlySpan<int> tokens = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);

        bool ok;
        fixed (int* tokPtr = tokens)
        {
            ok = NativeMethods.llama_state_save_file(
                _handle.DangerousHandle, path, tokPtr, (nuint)tokens.Length);
        }
        if (!ok)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_state_save_file),
                $"Failed to save state to '{path}'.");
        }
    }

    /// <summary>
    /// Load a context snapshot previously written by
    /// <see cref="SaveStateToFile"/>. Returns the token array that was saved
    /// alongside (empty if none were recorded).
    /// </summary>
    /// <param name="path">File written by <see cref="SaveStateToFile"/>.</param>
    /// <param name="maxTokenCapacity">
    /// Upper bound on the token array we're willing to allocate. Must be
    /// large enough to hold the tokens saved alongside the snapshot; if the
    /// saved array is larger, the native call will fail.
    /// </param>
    public unsafe int[] LoadStateFromFile(string path, int maxTokenCapacity = 65536)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokenCapacity);

        var buffer = new int[maxTokenCapacity];
        nuint count;
        bool ok;
        fixed (int* bufPtr = buffer)
        {
            ok = NativeMethods.llama_state_load_file(
                _handle.DangerousHandle, path, bufPtr, (nuint)maxTokenCapacity, &count);
        }
        if (!ok)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_state_load_file),
                $"Failed to load state from '{path}'. File may be missing, corrupt, " +
                $"from an incompatible llama.cpp version, or contain more than " +
                $"{maxTokenCapacity} tokens.");
        }
        if (count == 0) return Array.Empty<int>();
        var tokens = new int[(int)count];
        Array.Copy(buffer, tokens, (int)count);
        return tokens;
    }

    /// <summary>
    /// Size in bytes required to snapshot a single sequence's state.
    /// </summary>
    public long GetSequenceStateSize(int sequenceId, LlamaStateSeqFlags flags = LlamaStateSeqFlags.None)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return flags == LlamaStateSeqFlags.None
            ? (long)NativeMethods.llama_state_seq_get_size(_handle.DangerousHandle, sequenceId)
            : (long)NativeMethods.llama_state_seq_get_size_ext(_handle.DangerousHandle, sequenceId, (uint)flags);
    }

    /// <summary>
    /// Snapshot the state of a single sequence into a freshly-allocated array.
    /// Useful for cheap branching (clone a conversation, mutate one side,
    /// restore the other) without copying the rest of the KV cache.
    /// </summary>
    public byte[] SaveSequenceState(int sequenceId, LlamaStateSeqFlags flags = LlamaStateSeqFlags.None)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var size = GetSequenceStateSize(sequenceId, flags);
        var buffer = new byte[size];
        var written = SaveSequenceState(sequenceId, buffer, flags);
        if (written != buffer.Length)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_state_seq_get_data),
                $"Expected {buffer.Length} bytes, got {written}.");
        }
        return buffer;
    }

    /// <summary>
    /// Snapshot a single sequence's state into <paramref name="destination"/>.
    /// </summary>
    /// <returns>Number of bytes written.</returns>
    public unsafe int SaveSequenceState(
        int sequenceId, Span<byte> destination, LlamaStateSeqFlags flags = LlamaStateSeqFlags.None)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var required = (nuint)GetSequenceStateSize(sequenceId, flags);
        if ((nuint)destination.Length < required)
        {
            throw new ArgumentException(
                $"Destination buffer too small: need {required} bytes, have {destination.Length}.",
                nameof(destination));
        }
        fixed (byte* ptr = destination)
        {
            var written = flags == LlamaStateSeqFlags.None
                ? NativeMethods.llama_state_seq_get_data(
                    _handle.DangerousHandle, ptr, (nuint)destination.Length, sequenceId)
                : NativeMethods.llama_state_seq_get_data_ext(
                    _handle.DangerousHandle, ptr, (nuint)destination.Length, sequenceId, (uint)flags);
            return checked((int)written);
        }
    }

    /// <summary>
    /// Restore a single sequence's state from <paramref name="source"/>,
    /// writing into <paramref name="destinationSequenceId"/>. The source
    /// sequence ID captured at save time is ignored — the caller chooses
    /// the target slot, which makes this the primitive for cheap forks
    /// (save sequence 0, restore into sequence 1, diverge from there).
    /// </summary>
    public unsafe int RestoreSequenceState(
        int destinationSequenceId,
        ReadOnlySpan<byte> source,
        LlamaStateSeqFlags flags = LlamaStateSeqFlags.None)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.IsEmpty)
        {
            throw new ArgumentException("Snapshot is empty.", nameof(source));
        }
        fixed (byte* ptr = source)
        {
            var read = flags == LlamaStateSeqFlags.None
                ? NativeMethods.llama_state_seq_set_data(
                    _handle.DangerousHandle, ptr, (nuint)source.Length, destinationSequenceId)
                : NativeMethods.llama_state_seq_set_data_ext(
                    _handle.DangerousHandle, ptr, (nuint)source.Length, destinationSequenceId, (uint)flags);
            if (read == 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_state_seq_set_data),
                    "Failed to restore sequence state. Snapshot may be corrupt or incompatible.");
            }
            return checked((int)read);
        }
    }

    /// <summary>
    /// Save a single sequence's state plus its associated token array to
    /// <paramref name="path"/>.
    /// </summary>
    public unsafe void SaveSequenceStateToFile(
        string path, int sequenceId, ReadOnlySpan<int> tokens = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);

        nuint written;
        fixed (int* tokPtr = tokens)
        {
            written = NativeMethods.llama_state_seq_save_file(
                _handle.DangerousHandle, path, sequenceId, tokPtr, (nuint)tokens.Length);
        }
        if (written == 0)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_state_seq_save_file),
                $"Failed to save sequence {sequenceId} state to '{path}'.");
        }
    }

    /// <summary>
    /// Load a sequence snapshot previously written by
    /// <see cref="SaveSequenceStateToFile"/>, writing the KV state into
    /// <paramref name="destinationSequenceId"/>. Returns the tokens that
    /// were saved alongside (empty if none).
    /// </summary>
    public unsafe int[] LoadSequenceStateFromFile(
        string path, int destinationSequenceId, int maxTokenCapacity = 65536)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(path);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokenCapacity);

        var buffer = new int[maxTokenCapacity];
        nuint count;
        nuint read;
        fixed (int* bufPtr = buffer)
        {
            read = NativeMethods.llama_state_seq_load_file(
                _handle.DangerousHandle, path, destinationSequenceId,
                bufPtr, (nuint)maxTokenCapacity, &count);
        }
        if (read == 0)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_state_seq_load_file),
                $"Failed to load sequence state from '{path}'.");
        }
        if (count == 0) return Array.Empty<int>();
        var tokens = new int[(int)count];
        Array.Copy(buffer, tokens, (int)count);
        return tokens;
    }

    // ----- LoRA adapters (Tier 2 expansion) -----
    //
    // The pinned llama.cpp exposes a single `llama_set_adapters_lora` that
    // replaces the full active-adapter list in one call. We track the
    // desired set in managed memory and re-sync on every Attach/Detach so
    // callers can work in the more intuitive per-adapter model. Scales are
    // stored alongside the adapter pointer.
    //
    // The dictionary is keyed by adapter identity (reference equality). An
    // adapter attached twice with different scales gets its scale updated
    // in-place — the underlying native list only ever contains each adapter
    // at most once.

    private readonly Dictionary<LlamaLoraAdapter, float> _activeAdapters = new();

    /// <summary>
    /// Snapshot of the currently attached adapters and their scales. The
    /// returned dictionary is a copy — later mutations via
    /// <see cref="AttachLoraAdapter"/> / <see cref="DetachLoraAdapter"/> do
    /// not affect it.
    /// </summary>
    public IReadOnlyDictionary<LlamaLoraAdapter, float> ActiveLoraAdapters
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return new Dictionary<LlamaLoraAdapter, float>(_activeAdapters);
        }
    }

    /// <summary>
    /// Attach <paramref name="adapter"/> with the given <paramref name="scale"/>
    /// (0 = off, 1 = full, &gt;1 = amplified, &lt;0 = inverted). If the
    /// adapter is already attached, its scale is updated in place.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The adapter belongs to a different base model than this context.
    /// </exception>
    public void AttachLoraAdapter(LlamaLoraAdapter adapter, float scale = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(adapter);
        if (!ReferenceEquals(adapter.BaseModel, _model))
        {
            throw new ArgumentException(
                "LoRA adapter was loaded against a different base model than this context. " +
                "Adapters are not transferable between models — reload the adapter against " +
                "this context's model.",
                nameof(adapter));
        }

        _activeAdapters[adapter] = scale;
        SyncLoraAdapters();
    }

    /// <summary>
    /// Detach <paramref name="adapter"/> from this context. No-op if the
    /// adapter wasn't attached.
    /// </summary>
    public void DetachLoraAdapter(LlamaLoraAdapter adapter)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(adapter);
        if (_activeAdapters.Remove(adapter))
        {
            SyncLoraAdapters();
        }
    }

    /// <summary>
    /// Detach every adapter currently attached to this context.
    /// </summary>
    public void DetachAllLoraAdapters()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_activeAdapters.Count == 0) return;
        _activeAdapters.Clear();
        SyncLoraAdapters();
    }

    /// <summary>
    /// Atomic bulk replace of the active-adapter set. Useful when swapping
    /// from one adapter list to another without intermediate states. Pairs
    /// from <paramref name="adapters"/> with a scale of zero are dropped.
    /// </summary>
    public void SetLoraAdapters(IEnumerable<KeyValuePair<LlamaLoraAdapter, float>> adapters)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(adapters);

        _activeAdapters.Clear();
        foreach (var pair in adapters)
        {
            ArgumentNullException.ThrowIfNull(pair.Key);
            if (!ReferenceEquals(pair.Key.BaseModel, _model))
            {
                throw new ArgumentException(
                    "LoRA adapter was loaded against a different base model than this context.",
                    nameof(adapters));
            }
            _activeAdapters[pair.Key] = pair.Value;
        }
        SyncLoraAdapters();
    }

    private unsafe void SyncLoraAdapters()
    {
        int count = _activeAdapters.Count;
        if (count == 0)
        {
            // Clear path: llama.cpp accepts a NULL adapter array with n=0.
            var rc0 = NativeMethods.llama_set_adapters_lora(
                _handle.DangerousHandle, null, 0, null);
            if (rc0 != 0)
            {
                throw new LlamaException(nameof(NativeMethods.llama_set_adapters_lora), rc0,
                    "Failed to clear LoRA adapters on the context.");
            }
            return;
        }

        var ptrs = new IntPtr[count];
        var scales = new float[count];
        int i = 0;
        foreach (var kvp in _activeAdapters)
        {
            ptrs[i] = kvp.Key.Handle.DangerousHandle;
            scales[i] = kvp.Value;
            i++;
        }

        int rc;
        fixed (IntPtr* ptrPtr = ptrs)
        fixed (float* scalePtr = scales)
        {
            rc = NativeMethods.llama_set_adapters_lora(
                _handle.DangerousHandle, ptrPtr, (nuint)count, scalePtr);
        }
        if (rc != 0)
        {
            throw new LlamaException(nameof(NativeMethods.llama_set_adapters_lora), rc,
                $"llama_set_adapters_lora failed with status {rc} while syncing " +
                $"{count} adapter(s). Check the native log.");
        }
    }

    // ----- Multi-session support (Tier 3 / issue #5) -----
    //
    // The native context reserves a fixed number of sequence-id slots
    // (MaxSequenceCount, set at context creation). CreateSession hands out
    // the next free slot so callers can run isolated conversations against
    // the same shared model weights. The lock below serializes the one API
    // that can't be called concurrently on a single context: llama_decode.
    // Callers that stick to a single session per thread don't notice it;
    // callers that drive two sessions from two threads have their decodes
    // cooperatively interleaved rather than corrupting each other.

    private readonly HashSet<int> _liveSessions = new();
    private readonly System.Threading.SemaphoreSlim _decodeLock = new(1, 1);

    /// <summary>
    /// Allocate a new <see cref="LlamaSession"/> backed by the next free
    /// sequence slot in this context. Throws if every slot (up to
    /// <see cref="MaxSequenceCount"/>) is already in use — sessions must
    /// be disposed to return their slot to the pool.
    /// </summary>
    /// <remarks>
    /// <para>When <see cref="MaxSequenceCount"/> is 1 the only available
    /// slot is <c>seq_id = 0</c>, which is also the slot the legacy
    /// <see cref="LlamaGenerator(LlamaContext, LlamaSampler)"/> ctor targets
    /// implicitly. Creating a session there will succeed once, but running
    /// the legacy generator at the same time would step on the session's
    /// KV state. For multi-session workloads, set
    /// <see cref="LlamaContextParameters.MaxSequenceCount"/> &gt; 1 up front.</para>
    /// </remarks>
    public LlamaSession CreateSession()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_liveSessions)
        {
            for (int id = 0; id < MaxSequenceCount; id++)
            {
                if (_liveSessions.Add(id))
                {
                    return new LlamaSession(this, id);
                }
            }
        }
        throw new InvalidOperationException(
            $"All {MaxSequenceCount} sequence slots are in use. Dispose an existing " +
            "LlamaSession before creating another, or rebuild the context with a higher " +
            "LlamaContextParameters.MaxSequenceCount.");
    }

    internal void ReleaseSession(LlamaSession session)
    {
        if (session is null) return;
        lock (_liveSessions)
        {
            _liveSessions.Remove(session.SequenceId);
        }
    }

    /// <summary>
    /// Internal gate around calls that mutate the native context's shared
    /// logits buffer — i.e. <c>llama_decode</c> and any sampler call that
    /// reads logits produced by that decode. Generator loops acquire this
    /// for the decode+sample pair so another session's decode can't
    /// overwrite the logits mid-flight.
    /// </summary>
    internal async Task WithDecodeLockAsync(Action action, CancellationToken cancellationToken)
    {
        await _decodeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { action(); }
        finally { _decodeLock.Release(); }
    }

    /// <summary>
    /// Typed variant of <see cref="WithDecodeLockAsync(Action, CancellationToken)"/>
    /// for calls that compute a value under the lock (e.g. sample + decode →
    /// returned token id).
    /// </summary>
    internal async Task<T> WithDecodeLockAsync<T>(Func<T> action, CancellationToken cancellationToken)
    {
        await _decodeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { return action(); }
        finally { _decodeLock.Release(); }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _memoryHandleCache = IntPtr.Zero;
        _activeAdapters.Clear();
        lock (_liveSessions) { _liveSessions.Clear(); }
        _decodeLock.Dispose();
        _handle.Dispose();
    }
}
