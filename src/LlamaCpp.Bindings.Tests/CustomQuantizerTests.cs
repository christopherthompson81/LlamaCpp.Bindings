namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Algorithm-level tests for <see cref="LlamaCustomQuantizer"/>'s
/// guardrail logic. Full end-to-end tests against a real GGUF live in
/// the smoke harness (require a model file present) — these are the
/// pieces we can test without a binary asset.
/// </summary>
public class CustomQuantizerTests
{
    static CustomQuantizerTests()
    {
        LlamaBackend.Initialize();
    }

    [Theory]
    [InlineData(LlamaTensorType.Q4_K, 256, LlamaTensorType.Q4_K)]    // ncols divisible by 256: no fallback
    [InlineData(LlamaTensorType.Q4_K, 100, LlamaTensorType.Q5_0)]    // ncols not divisible: legacy fallback
    [InlineData(LlamaTensorType.Q5_K, 100, LlamaTensorType.Q5_1)]
    [InlineData(LlamaTensorType.Q6_K, 100, LlamaTensorType.Q8_0)]
    [InlineData(LlamaTensorType.Q2_K, 100, LlamaTensorType.Q4_0)]
    [InlineData(LlamaTensorType.Q3_K, 100, LlamaTensorType.Q4_0)]
    [InlineData(LlamaTensorType.IQ4_XS, 100, LlamaTensorType.IQ4_NL)]
    public void ShapeFallback_MapsKQuantsToLegacyTypes(
        LlamaTensorType target, long ncols, LlamaTensorType expected)
    {
        // Reflect into the private ApplyShapeFallback to avoid needing
        // a full GGUF; the table is the contract worth locking down.
        var method = typeof(LlamaCustomQuantizer).GetMethod(
            "ApplyShapeFallback",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        var tensorInfoCtor = typeof(LlamaGgufTensorInfo).GetConstructors().Single();
        var info = (LlamaGgufTensorInfo)tensorInfoCtor.Invoke(new object[]
        {
            "test.weight", (uint)1, new long[] { ncols, 1 }, ncols * 2L, 0L
        });
        var result = (LlamaTensorType)method.Invoke(null, new object[] { info, target })!;
        Assert.Equal(expected, result);
    }
}
