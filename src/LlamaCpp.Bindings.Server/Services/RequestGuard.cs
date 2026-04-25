using LlamaCpp.Bindings.Server.Configuration;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Cross-cutting request-time guards: prompt-length cap (413 protection)
/// and server-side request timeout. Centralised so chat + completion +
/// any future endpoint share the same defaults and the same precedence
/// rules.
/// </summary>
public static class RequestGuard
{
    /// <summary>
    /// Resolve the effective prompt-length ceiling for a request. When
    /// the operator left <see cref="ServerOptions.MaxPromptTokens"/> at
    /// 0 we derive it from <c>ContextSize - MaxOutputTokens</c> so the
    /// reply always fits; an explicit positive value wins.
    /// </summary>
    public static int EffectiveMaxPromptTokens(ServerOptions opts, int contextSize)
    {
        if (opts.MaxPromptTokens > 0) return opts.MaxPromptTokens;
        // Floor of 1 protects against the degenerate case where MaxOutputTokens
        // is set absurdly high relative to ContextSize.
        return Math.Max(1, contextSize - opts.MaxOutputTokens);
    }

    /// <summary>
    /// Build a cancellation token linked to <paramref name="requestAborted"/>
    /// (the client side) AND a freshly-created timeout source (the
    /// server side). Returns the combined disposable + token. Null
    /// timeout source means timeouts are disabled
    /// (<c>RequestTimeoutSeconds &lt;= 0</c>); the caller still gets a
    /// CTS for cleanup.
    /// </summary>
    public static (CancellationTokenSource Linked, CancellationTokenSource? Timeout)
        CreateLinkedToken(CancellationToken requestAborted, ServerOptions opts)
    {
        CancellationTokenSource? timeoutCts = null;
        if (opts.RequestTimeoutSeconds > 0)
        {
            timeoutCts = new CancellationTokenSource(
                TimeSpan.FromSeconds(opts.RequestTimeoutSeconds));
        }
        var linked = timeoutCts is null
            ? CancellationTokenSource.CreateLinkedTokenSource(requestAborted)
            : CancellationTokenSource.CreateLinkedTokenSource(requestAborted, timeoutCts.Token);
        return (linked, timeoutCts);
    }
}
