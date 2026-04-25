using System.Text;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Streaming-safe scanner that watches the generator's output for any of a
/// caller-supplied set of stop strings. The natural shape — "check if the
/// emitted text ends with any stop" — is wrong in isolation: a stop string
/// can straddle token boundaries (e.g. <c>"Human:"</c> arriving as
/// <c>"Hu"</c> + <c>"man:"</c>), so emitting tokens the moment they arrive
/// would ship bytes we'd want to retract once the match completes.
/// </summary>
/// <remarks>
/// <para>The matcher solves that by holding back the tail of the output —
/// up to <c>max(stopLen) - 1</c> characters — until enough subsequent text
/// has arrived to rule out a cross-boundary partial match. When a stop
/// fires, the accumulated text is truncated to exclude the stop string and
/// the caller receives exactly the bytes it should stream. On a clean end
/// of generation (EOG / max tokens), the caller drains the held-back tail
/// via <see cref="Flush"/>.</para>
///
/// <para>Operates on C# strings (UTF-16 code units). Upstream
/// <c>LlamaGenerator</c> emits pieces at UTF-8 character boundaries
/// already, so partial-codepoint concerns don't reach this layer.</para>
/// </remarks>
public sealed class StopMatcher
{
    private readonly string[] _stops;
    private readonly int _holdBack;
    private readonly StringBuilder _accumulated = new();
    private int _emittedLength;

    /// <summary>Empty stop-set: the matcher is a pass-through.</summary>
    public bool IsEmpty => _stops.Length == 0;

    public StopMatcher(IReadOnlyList<string>? stops)
    {
        if (stops is null)
        {
            _stops = Array.Empty<string>();
            _holdBack = 0;
            return;
        }
        _stops = stops.Where(s => !string.IsNullOrEmpty(s)).ToArray();
        _holdBack = _stops.Length == 0 ? 0 : _stops.Max(s => s.Length) - 1;
    }

    /// <summary>Feed a newly-produced piece from the generator.</summary>
    /// <returns>
    /// The bytes the caller should emit right now (may be empty) and a
    /// flag indicating whether a stop just fired. When <c>Stopped</c> is
    /// true, the returned string is the complete output up to (but not
    /// including) the matched stop string, minus whatever was emitted by
    /// prior <see cref="Offer"/> calls. The caller must stop feeding the
    /// matcher once <c>Stopped</c> is true.
    /// </returns>
    public (string Emit, bool Stopped) Offer(string piece)
    {
        if (IsEmpty)
        {
            // Fast path when there are no stops — always pass through.
            _accumulated.Append(piece);
            _emittedLength += piece.Length;
            return (piece, false);
        }

        _accumulated.Append(piece);
        var acc = _accumulated.ToString();

        // A stop can land mid-piece (a single decoded piece like " C D E"
        // might contain " C " in its middle), so we can't just check the
        // tail — we have to scan the entire not-yet-emitted region. Walk
        // each stop string starting from the last emitted position and
        // take the earliest hit.
        int earliestMatch = int.MaxValue;
        int matchLen = 0;
        foreach (var s in _stops)
        {
            int idx = acc.IndexOf(s, _emittedLength, StringComparison.Ordinal);
            if (idx >= 0 && idx < earliestMatch)
            {
                earliestMatch = idx;
                matchLen = s.Length;
            }
        }

        if (earliestMatch != int.MaxValue)
        {
            // Match fired. Emit everything before the stop, advance the
            // emitted-length cursor past the stop string itself so Flush
            // won't re-emit it.
            string emit = acc.Substring(_emittedLength, earliestMatch - _emittedLength);
            _emittedLength = earliestMatch + matchLen;
            return (emit, true);
        }

        // No full stop matched. A partial could still be forming at the
        // tail of the accumulation — hold back the last `_holdBack`
        // characters so we can see whether the NEXT piece completes a
        // stop that straddles this boundary. Emit everything older.
        int safeLen = Math.Max(0, acc.Length - _holdBack);
        if (safeLen > _emittedLength)
        {
            string emit = acc.Substring(_emittedLength, safeLen - _emittedLength);
            _emittedLength = safeLen;
            return (emit, false);
        }
        return (string.Empty, false);
    }

    /// <summary>
    /// Drain whatever's been held back waiting for a possible stop match.
    /// Call when generation ends for a reason other than a stop hit
    /// (EOG, max tokens, cancellation).
    /// </summary>
    public string Flush()
    {
        if (_accumulated.Length <= _emittedLength) return string.Empty;
        string emit = _accumulated.ToString(_emittedLength, _accumulated.Length - _emittedLength);
        _emittedLength = _accumulated.Length;
        return emit;
    }
}
