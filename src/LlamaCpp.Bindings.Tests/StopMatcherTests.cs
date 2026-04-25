using LlamaCpp.Bindings.Server.Services;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Unit tests for <see cref="StopMatcher"/>. No model required — these
/// exercise the pure-managed hold-back / match logic that the endpoints
/// rely on. The integration tests in <c>LlamaServerTests</c> verify the
/// end-to-end wiring; these catch logic regressions quickly.
/// </summary>
public class StopMatcherTests
{
    [Fact]
    public void Empty_Stop_Set_Is_Passthrough()
    {
        var m = new StopMatcher(null);
        Assert.True(m.IsEmpty);
        var (emit, stopped) = m.Offer("hello world");
        Assert.Equal("hello world", emit);
        Assert.False(stopped);
        Assert.Equal("", m.Flush()); // nothing held back
    }

    [Fact]
    public void Single_Stop_Fires_On_Suffix_Match_And_Strips_Stop()
    {
        var m = new StopMatcher(new[] { "END" });
        var (e1, s1) = m.Offer("the ");   // 4 chars; hold back 2 (len-1)
        var (e2, s2) = m.Offer("END");    // "the END" — ends with "END"

        // e1 may be empty or "th" depending on hold-back; core invariant
        // is that by the time we're done, we've emitted exactly "the "
        // and no more.
        Assert.False(s1);
        Assert.True(s2);
        Assert.Equal("the ", e1 + e2);
    }

    [Fact]
    public void Stop_Split_Across_Two_Offers_Still_Matches()
    {
        // Regression guard: this is the primary reason the matcher exists.
        // If the endpoint emitted each piece as it arrived, "Hu" would
        // land on the wire before "man:" fired the match.
        var m = new StopMatcher(new[] { "Human:" });
        var (e1, s1) = m.Offer("Hello Hu");
        var (e2, s2) = m.Offer("man:");

        Assert.False(s1);
        Assert.True(s2);
        Assert.Equal("Hello ", e1 + e2);
    }

    [Fact]
    public void Hold_Back_Keeps_Tail_Until_Safe_To_Emit()
    {
        // "Hu" alone COULD be the start of "Human:". With hold_back = 5
        // (len("Human:") - 1), "Hu" stays buffered until the next offer
        // makes it clear nothing stop-like is pending.
        var m = new StopMatcher(new[] { "Human:" });
        var (e1, _) = m.Offer("Hu");
        Assert.Equal("", e1); // buffered, not yet safe

        var (e2, _) = m.Offer(" there, my friend.");
        // By now the accumulated string is long enough that "Hu" is past
        // the tail hold-back window and must have been emitted.
        Assert.StartsWith("Hu", e1 + e2);
        Assert.DoesNotContain("Human:", e1 + e2);
    }

    [Fact]
    public void Flush_Drains_Held_Back_Bytes_When_Generation_Ends_Without_Match()
    {
        // If the model hits EOG / max_tokens without tripping a stop,
        // the endpoint must call Flush() or the last few characters
        // disappear silently.
        var m = new StopMatcher(new[] { "STOPSTRING" }); // len 10, hold 9
        var (e, _) = m.Offer("short");   // all held
        Assert.Equal("", e);
        var flushed = m.Flush();
        Assert.Equal("short", flushed);
    }

    [Fact]
    public void Multiple_Stops_First_Hit_Wins()
    {
        var m = new StopMatcher(new[] { "A", "BB" });
        // After "BB" both could match at different tail positions, but we
        // only care that a match fires and we stop.
        var (emit, stopped) = m.Offer("BB");
        Assert.True(stopped);
        // "A" would truncate at 0; "BB" would truncate at 0. Either way
        // no content should be emitted.
        Assert.Equal("", emit);
    }

    [Fact]
    public void Stop_In_Middle_Of_Output_Fires_When_Last_Char_Arrives()
    {
        // Output "foo STOP bar" with stop "STOP": we should emit "foo "
        // and terminate; "bar" never gets sampled because the endpoint
        // breaks out of the loop.
        var m = new StopMatcher(new[] { "STOP" });
        var (e1, s1) = m.Offer("foo ");
        var (e2, s2) = m.Offer("STOP");

        Assert.False(s1);
        Assert.True(s2);
        Assert.Equal("foo ", e1 + e2);
    }

    [Fact]
    public void Empty_And_Null_Stops_Are_Ignored()
    {
        // Callers can send weird input shapes; the matcher should
        // quietly drop empty stops rather than match-on-anything.
        var m = new StopMatcher(new[] { "", "real" });
        var (_, s1) = m.Offer("hello ");
        var (_, s2) = m.Offer("real");
        Assert.False(s1);
        Assert.True(s2);
    }

    [Fact]
    public void Accumulated_Longer_Than_HoldBack_Emits_Early()
    {
        // Sanity check on the streaming invariant: a long run with no
        // match should flush most of its output incrementally, not only
        // at the end.
        var m = new StopMatcher(new[] { "END" }); // hold back 2 chars
        var (e1, _) = m.Offer("this is a longer string with no match yet");
        // 41 chars in, 2 held back = 39 emitted.
        Assert.Equal(39, e1.Length);
        Assert.StartsWith("this is a longer string", e1);
    }
}
