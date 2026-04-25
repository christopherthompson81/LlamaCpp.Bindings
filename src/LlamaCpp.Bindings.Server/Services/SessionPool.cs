namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Bounded pool that hands out <see cref="LlamaSession"/> leases to
/// concurrent HTTP requests. Limits in-flight generations to the
/// context's <see cref="LlamaContext.MaxSequenceCount"/>; beyond that,
/// requests queue on the semaphore rather than failing.
/// </summary>
/// <remarks>
/// <para>The binding's own <see cref="LlamaContext.CreateSession"/> already
/// rejects over-allocation, but it throws synchronously — not the right
/// shape for a server that should hold clients in a queue. We wrap it in
/// a semaphore whose permit count matches the pool.</para>
///
/// <para>The lease is <see cref="IDisposable"/>; dispose returns both the
/// sequence slot (via the inner session) and the permit. Forgetting to
/// dispose eventually exhausts the pool.</para>
/// </remarks>
public sealed class SessionPool : IDisposable
{
    private readonly ModelHost _host;
    private readonly SemaphoreSlim _permits;
    private bool _disposed;

    public SessionPool(ModelHost host)
    {
        _host = host;
        _permits = new SemaphoreSlim(host.Context.MaxSequenceCount, host.Context.MaxSequenceCount);
    }

    /// <summary>
    /// Wait for a free slot, then allocate a session. The returned lease
    /// must be disposed — typically via <c>await using</c>.
    /// </summary>
    public async Task<SessionLease> LeaseAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _permits.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var session = _host.Context.CreateSession();
            return new SessionLease(session, _permits);
        }
        catch
        {
            // CreateSession can still throw (race between permit acquire
            // and session allocation shouldn't happen given our bookkeeping,
            // but be defensive — always return the permit on failure).
            _permits.Release();
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _permits.Dispose();
    }
}

/// <summary>
/// One in-flight request's borrow of a <see cref="LlamaSession"/>. Dispose
/// returns the session's sequence slot to the pool.
/// </summary>
public sealed class SessionLease : IDisposable
{
    private readonly SemaphoreSlim _permits;
    private bool _disposed;

    public LlamaSession Session { get; }

    internal SessionLease(LlamaSession session, SemaphoreSlim permits)
    {
        Session = session;
        _permits = permits;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Session.Dispose();
        _permits.Release();
    }
}
