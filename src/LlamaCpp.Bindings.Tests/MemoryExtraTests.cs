namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tier-1 C5 coverage: shift, divide, breakdown-print.
/// Shares the GPU fixture with the other multi-turn tests.
/// </summary>
[Collection(GpuCollection.Name)]
public class MemoryExtraTests
{
    private readonly GpuGenerationFixture _fx;
    public MemoryExtraTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public void ShiftSequencePositions_Behaves_Per_Backend_Support()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        if (!_fx.Context.SupportsPositionShift())
        {
            // Qwen3.6 uses IMRope (multi-dimensional positions) which the
            // native KV cache refuses to shift. The binding turns the native
            // GGML_ASSERT into a clean NotSupportedException so callers don't
            // abort the process.
            Assert.Throws<NotSupportedException>(() =>
                _fx.Context.ShiftSequencePositions(0, 0, 1, 1));
            Assert.Throws<NotSupportedException>(() =>
                _fx.Context.DivideSequencePositions(0, 0, 1, 2));
            return;
        }

        // Shift-supported models (standard 1D RoPE — LLaMA, Mistral family).
        // Testing this branch requires loading a compatible GGUF; skip on
        // machines that only have the Qwen fixture.
    }

    [Fact]
    public void DivideSequencePositions_Rejects_NonPositive_Divisor()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _fx.Context.DivideSequencePositions(0, 0, -1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _fx.Context.DivideSequencePositions(0, 0, -1, -2));
    }

    [Fact]
    public void LogMemoryBreakdown_Runs_Without_Throwing()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        // Output goes to the native log sink (stderr / our Avalonia log
        // routing). We only verify the call doesn't throw — the content
        // isn't easily asserted from managed code.
        _fx.Context.LogMemoryBreakdown();
    }
}
