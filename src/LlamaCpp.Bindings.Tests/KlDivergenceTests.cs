namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Math sanity tests for <see cref="LlamaKlDivergence"/>. The most
/// load-bearing check is the same-model-twice case: KL of a
/// distribution against itself is identically zero, modulo floating-
/// point noise. If our log-sum-exp / KL accumulator is off, this
/// fires immediately. We also assert that top-1/top-5 agreement is
/// exactly 1.0 in that case.
/// </summary>
public class KlDivergenceTests
{
    [Fact]
    public async Task Same_Model_Twice_Has_Zero_KL_And_Perfect_Agreement()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        // Load the same model twice — different LlamaModel/LlamaContext
        // instances but identical weights, so per-position
        // distributions match bit-for-bit and KL collapses to 0.
        using var refModel = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });
        using var testModel = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var corpus =
            "The quick brown fox jumps over the lazy dog. " +
            "Pack my box with five dozen liquor jugs. " +
            "How vexingly quick daft zebras jump.";

        var result = await LlamaKlDivergence.ComputeAsync(
            refModel, testModel, corpus,
            new LlamaKlDivergenceOptions
            {
                ContextSize = 64,
                ScoreSecondHalfOnly = false,
            });

        Assert.True(result.TokensScored > 0);
        // KL of P||P is 0; with float32 logits and double-precision
        // accumulation the result should be tiny. 1e-4 nats is
        // generously loose so the check is robust to non-determinism
        // in CPU and GPU backends.
        Assert.True(result.MeanKl < 1e-4,
            $"Same-model-twice mean KL should be ~0, got {result.MeanKl}.");
        Assert.True(result.MaxKl < 1e-3,
            $"Same-model-twice max KL should be ~0, got {result.MaxKl}.");

        // Argmax agrees by construction.
        Assert.Equal(1.0, result.Top1AgreementRate, 6);
        Assert.Equal(1.0, result.Top5AgreementRate, 6);

        // Both perplexities should be identical.
        Assert.Equal(result.ReferencePerplexity, result.TestPerplexity, 6);
    }

    [Fact]
    public async Task Honors_Cancellation_Between_Chunks()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var refModel = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });
        using var testModel = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            LlamaKlDivergence.ComputeAsync(
                refModel, testModel,
                "The quick brown fox jumps over the lazy dog.",
                new LlamaKlDivergenceOptions { ContextSize = 32 },
                progress: null,
                cancellationToken: cts.Token));
    }
}
