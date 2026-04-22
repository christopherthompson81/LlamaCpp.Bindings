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
    private IntPtr Memory()
    {
        if (_memoryHandleCache == IntPtr.Zero)
        {
            _memoryHandleCache = NativeMethods.llama_get_memory(_handle.DangerousHandle);
            if (_memoryHandleCache == IntPtr.Zero)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_get_memory),
                    "llama_get_memory returned NULL — this context has no attached memory.");
            }
        }
        return _memoryHandleCache;
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _memoryHandleCache = IntPtr.Zero;
        _handle.Dispose();
    }
}
