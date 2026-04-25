namespace LlamaCpp.Bindings;

/// <summary>
/// A handle to one conversation slot inside a shared <see cref="LlamaContext"/>.
/// Each session owns an exclusive <c>seq_id</c> in the context's KV cache, so
/// multiple sessions can run concurrent conversations against the same loaded
/// model without their histories bleeding into each other.
/// </summary>
/// <remarks>
/// <para><b>Creation:</b> obtain via <see cref="LlamaContext.CreateSession"/>.
/// The context allocates the next free sequence id out of its
/// <see cref="LlamaContext.MaxSequenceCount"/> slots; if every slot is live
/// the call throws rather than overwriting someone else's history.</para>
///
/// <para><b>Lifetime:</b> <see cref="IDisposable"/>. Disposing a session
/// clears its KV range and returns the sequence id to the pool — the next
/// <see cref="LlamaContext.CreateSession"/> may reuse it. Forgetting to
/// dispose is not a memory leak (the KV slots still live with the context)
/// but permanently consumes a slot in the pool until the owning context is
/// itself disposed.</para>
///
/// <para><b>Thread-safety:</b> calls that touch the native context (decode,
/// sample, KV ops) serialize via a lock inside <see cref="LlamaContext"/> —
/// llama.cpp's own contract forbids concurrent <c>llama_decode</c> on one
/// context. Sessions are the unit of concurrency: run each session in its
/// own task, and the generator loops will cooperatively interleave. A
/// single session's operations must still be driven from one thread at a
/// time (generator loops are not re-entrant).</para>
///
/// <para><b>Pairing with <see cref="LlamaGenerator"/>:</b> a session holds
/// the seq_id and history bookkeeping; a generator holds the sampler and
/// the streaming decode loop. Use the
/// <see cref="LlamaGenerator(LlamaSession, LlamaSampler)"/> ctor to bind
/// them. The legacy <see cref="LlamaGenerator(LlamaContext, LlamaSampler)"/>
/// continues to use seq_id 0 (the implicit session for single-conversation
/// contexts).</para>
/// </remarks>
public sealed class LlamaSession : IDisposable
{
    private readonly LlamaContext _owner;
    private bool _disposed;

    /// <summary>Sequence id assigned to this session by its owning context.</summary>
    public int SequenceId { get; }

    /// <summary>The context this session lives in.</summary>
    public LlamaContext Context
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _owner;
        }
    }

    internal LlamaSession(LlamaContext owner, int sequenceId)
    {
        _owner = owner;
        SequenceId = sequenceId;
    }

    /// <summary>
    /// Drop every token recorded for this session from the KV cache without
    /// releasing its slot. Useful for starting a new conversation on the
    /// same session id while keeping any per-session bookkeeping you may
    /// have layered on top.
    /// </summary>
    public void ClearHistory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _owner.RemoveSequenceRange(SequenceId, fromPosition: 0, toPosition: -1);
    }

    /// <summary>
    /// Inclusive/exclusive position bounds currently recorded for this
    /// session (<c>(null, null)</c> if the session has not decoded anything
    /// yet). Shortcut over
    /// <see cref="LlamaContext.SequencePositionRange"/>.
    /// </summary>
    public (int? Minimum, int? Maximum) PositionRange
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _owner.SequencePositionRange(SequenceId);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Clear this session's KV range and return its slot to the pool.
        // Ignore errors from RemoveSequenceRange — best-effort cleanup; if
        // the backend refuses partial trim the slot can still be reused
        // after an explicit ClearKvCache(). We still release the slot so
        // the pool bookkeeping doesn't leak.
        try
        {
            _owner.RemoveSequenceRange(SequenceId, 0, -1);
        }
        catch
        {
            // swallowed: dispose must not throw
        }
        _owner.ReleaseSession(this);
    }
}
