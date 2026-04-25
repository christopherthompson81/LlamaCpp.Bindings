namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Bounded pool of <see cref="LlamaSession"/>s that persist across HTTP
/// requests so the server can reuse cached prompt prefixes — the classic
/// "follow-up chat turn doesn't re-decode the whole conversation" win.
/// </summary>
/// <remarks>
/// <para>Sessions are pre-allocated at construction (one per
/// <see cref="LlamaContext.MaxSequenceCount"/> slot) and held for the pool's
/// lifetime. Leasing a slot does <em>not</em> dispose its session; it hands
/// the slot to the caller with the KV state from its last use still intact.
/// When the caller returns it, the pool records the final token sequence
/// so the next request can match prefixes against it.</para>
///
/// <para>Matching picks the idle slot whose cached tokens share the longest
/// common prefix with the incoming request. Ties break to the least-
/// recently-used slot so older cache entries get displaced first. No match
/// just means LCP = 0 and we pick the LRU idle slot to overwrite.</para>
///
/// <para>The semaphore caps in-flight requests at
/// <see cref="LlamaContext.MaxSequenceCount"/>. Anything beyond queues on
/// the semaphore until a slot is returned.</para>
/// </remarks>
public sealed class SessionPool : IDisposable
{
    private readonly ModelHost _host;
    private readonly SemaphoreSlim _permits;
    private readonly Slot[] _slots;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>Per-slot bookkeeping. Internal so <see cref="SessionLease"/> can reference it.</summary>
    internal sealed class Slot
    {
        public required LlamaSession Session { get; init; }
        /// <summary>Tokens currently materialised in this slot's KV (prompt + generated from last lease).</summary>
        public int[] Tokens { get; set; } = Array.Empty<int>();
        /// <summary>Monotonic tick-count timestamp of the last release. Drives LRU tiebreaks.</summary>
        public long LastUsedTicks { get; set; }
        public bool InUse { get; set; }
    }

    public SessionPool(ModelHost host)
    {
        _host = host;
        int n = host.Context.MaxSequenceCount;
        _permits = new SemaphoreSlim(n, n);
        _slots = new Slot[n];
        for (int i = 0; i < n; i++)
        {
            _slots[i] = new Slot { Session = host.Context.CreateSession() };
        }
    }

    /// <summary>
    /// Count of pool slots. Fixed at construction to the context's
    /// <see cref="LlamaContext.MaxSequenceCount"/>.
    /// </summary>
    public int SlotCount => _slots.Length;

    /// <summary>
    /// Wait for a free slot, then allocate a lease that reuses the longest
    /// available cached prefix. The returned lease is
    /// <see cref="IDisposable"/>; dispose returns the slot to the pool
    /// with its tokens updated to reflect whatever the caller decoded.
    /// </summary>
    public async Task<SessionLease> LeaseAsync(int[] promptTokens, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(promptTokens);

        await _permits.WaitAsync(cancellationToken).ConfigureAwait(false);
        Slot picked;
        int firstNewIndex;
        try
        {
            lock (_lock)
            {
                Slot? best = null;
                int bestLcp = -1;
                long bestLru = long.MaxValue;
                foreach (var slot in _slots)
                {
                    if (slot.InUse) continue;
                    int match = CommonPrefixLength(slot.Tokens, promptTokens);
                    // Pick the slot with the largest common prefix. Tiebreak
                    // to LRU (lower LastUsedTicks) so warmer cache entries
                    // are preserved for potential future hits.
                    if (match > bestLcp ||
                        (match == bestLcp && slot.LastUsedTicks < bestLru))
                    {
                        best = slot;
                        bestLcp = match;
                        bestLru = slot.LastUsedTicks;
                    }
                }
                // At least one slot is free because we hold a permit.
                picked = best!;
                firstNewIndex = bestLcp;
                picked.InUse = true;
            }
        }
        catch
        {
            _permits.Release();
            throw;
        }

        // Trim anything past the common prefix. The slot's InUse flag blocks
        // other leases from this slot, so we can do this outside the pool
        // lock. RemoveSequenceRange mutates the KV layout but does not
        // decode, so it doesn't need the context's decode lock either.
        if (firstNewIndex < picked.Tokens.Length)
        {
            picked.Session.Context.RemoveSequenceRange(
                picked.Session.SequenceId,
                fromPosition: firstNewIndex,
                toPosition: -1);
        }

        return new SessionLease(this, picked, promptTokens, firstNewIndex);
    }

    internal void Return(Slot slot, int[] finalTokens)
    {
        lock (_lock)
        {
            slot.Tokens = finalTokens;
            slot.LastUsedTicks = Environment.TickCount64;
            slot.InUse = false;
        }
        _permits.Release();
    }

    /// <summary>Length of the longest prefix on which <paramref name="a"/> and <paramref name="b"/> agree.</summary>
    internal static int CommonPrefixLength(int[] a, int[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            if (a[i] != b[i]) return i;
        }
        return n;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var slot in _slots)
        {
            slot.Session.Dispose();
        }
        _permits.Dispose();
    }
}

/// <summary>
/// One in-flight request's borrow of a pool slot. Exposes the inner
/// <see cref="LlamaSession"/> plus metadata the generator needs to skip
/// the cached prefix (<see cref="FirstNewIndex"/>) and to record what
/// actually landed in KV on the way out (<see cref="OnTokenDecoded"/>).
/// </summary>
public sealed class SessionLease : IDisposable
{
    private readonly SessionPool _pool;
    private readonly SessionPool.Slot _slot;
    private readonly int[] _promptTokens;
    private List<int>? _generated;
    private bool _disposed;

    public LlamaSession Session { get; }

    /// <summary>
    /// Prompt tokens. Indices <c>[0, <see cref="FirstNewIndex"/>)</c> were
    /// already in KV before this lease; indices
    /// <c>[<see cref="FirstNewIndex"/>, Length)</c> need to be decoded.
    /// </summary>
    public int[] PromptTokens => _promptTokens;

    /// <summary>
    /// How many prompt tokens were cache-hits. Pass this to
    /// <c>LlamaGenerator.GenerateAsync(firstNewIndex: ...)</c> so the
    /// generator skips redecoding them.
    /// </summary>
    public int FirstNewIndex { get; }

    /// <summary>Cached tokens ÷ total prompt tokens, as a percentage.</summary>
    public int CachedTokens => FirstNewIndex;

    internal SessionLease(SessionPool pool, SessionPool.Slot slot, int[] promptTokens, int firstNewIndex)
    {
        _pool = pool;
        _slot = slot;
        _promptTokens = promptTokens;
        Session = slot.Session;
        FirstNewIndex = firstNewIndex;
    }

    /// <summary>
    /// Pass as the <c>onTokenDecoded</c> callback to
    /// <see cref="LlamaGenerator.GenerateAsync(IReadOnlyList{int}, int, bool, int, Action{int}, CancellationToken)"/>
    /// so the lease can track which generated tokens made it into KV.
    /// </summary>
    public void OnTokenDecoded(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        (_generated ??= new List<int>()).Add(token);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        int[] finalTokens;
        if (_generated is null || _generated.Count == 0)
        {
            finalTokens = _promptTokens;
        }
        else
        {
            finalTokens = new int[_promptTokens.Length + _generated.Count];
            Array.Copy(_promptTokens, finalTokens, _promptTokens.Length);
            for (int i = 0; i < _generated.Count; i++)
            {
                finalTokens[_promptTokens.Length + i] = _generated[i];
            }
        }
        _pool.Return(_slot, finalTokens);
    }
}
