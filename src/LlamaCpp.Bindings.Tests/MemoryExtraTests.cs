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
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _fx.Context.DivideSequencePositions(0, 0, -1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _fx.Context.DivideSequencePositions(0, 0, -1, -2));
    }

    // Note: llama_memory_breakdown_print was removed upstream in the 2026-04
    // refactor. The public wrapper went away with it; this test was removed
    // to follow. See docs/mtmd_investigation.md Run 2.
}
