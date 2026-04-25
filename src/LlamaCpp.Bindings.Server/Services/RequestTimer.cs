using System.Diagnostics;
using LlamaCpp.Bindings.Server.Models;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Per-request stopwatch + token counter. Wraps the state the endpoints
/// need to fill out a <see cref="RequestTimings"/> object and to feed
/// the global <see cref="ServerMetrics"/> counters. Cheap — no
/// allocations on the hot path besides the stopwatch itself.
/// </summary>
/// <remarks>
/// <para>Timing breakdown:</para>
/// <list type="bullet">
///   <item><c>PromptMs</c> runs from construction to the first call to
///   <see cref="MarkFirstToken"/>; covers the prompt-decode wall clock.</item>
///   <item><c>PredictedMs</c> runs from <see cref="MarkFirstToken"/> to
///   <see cref="Finish"/>; covers the sampling loop.</item>
/// </list>
/// Both timers are wall-clock — cooperative cancellation / semaphore
/// waits count — which matches what llama-server reports and what
/// clients care about.
/// </remarks>
public sealed class RequestTimer
{
    private readonly Stopwatch _overall = Stopwatch.StartNew();
    private long _firstTokenTicks = -1;
    private long _finishTicks = -1;

    public int PromptTokens { get; }
    public int CachedTokens { get; }
    public int PredictedTokens { get; private set; }

    public RequestTimer(int promptTokensToDecode, int cachedTokens)
    {
        PromptTokens = promptTokensToDecode;
        CachedTokens = cachedTokens;
    }

    /// <summary>Call once, when the first generated token leaves the sampler.</summary>
    public void MarkFirstToken()
    {
        if (_firstTokenTicks < 0) _firstTokenTicks = _overall.ElapsedTicks;
    }

    public void IncrementPredicted() => PredictedTokens++;

    public void Finish()
    {
        if (_finishTicks < 0) _finishTicks = _overall.ElapsedTicks;
    }

    /// <summary>
    /// Produce the <see cref="RequestTimings"/> payload for the response
    /// body. Assumes <see cref="Finish"/> has already been called; if it
    /// hasn't, the current elapsed time stands in.
    /// </summary>
    public RequestTimings Snapshot()
    {
        var end = _finishTicks >= 0 ? _finishTicks : _overall.ElapsedTicks;
        var firstTok = _firstTokenTicks >= 0 ? _firstTokenTicks : end;

        double promptMs = TicksToMs(firstTok);
        double predictedMs = TicksToMs(end - firstTok);

        return new RequestTimings
        {
            PromptN = PromptTokens,
            PromptMs = promptMs,
            PromptPerTokenMs = PromptTokens > 0 ? promptMs / PromptTokens : 0,
            PredictedN = PredictedTokens,
            PredictedMs = predictedMs,
            PredictedPerTokenMs = PredictedTokens > 0 ? predictedMs / PredictedTokens : 0,
            CachedN = CachedTokens,
        };
    }

    private static double TicksToMs(long ticks) =>
        ticks * 1000.0 / Stopwatch.Frequency;
}
