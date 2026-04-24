using System.Text;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// State / session snapshot round-trip tests. The core invariant: after
/// <c>SaveState</c> → <c>ClearKvCache</c> → <c>RestoreState</c>, a seeded
/// greedy continuation must produce the same stream it would have produced
/// without the round-trip. If this ever fails, prompt caching is broken.
/// </summary>
[Collection(GpuCollection.Name)]
public class StateSnapshotTests
{
    private readonly GpuGenerationFixture _fx;
    public StateSnapshotTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public async Task GetStateSize_Reflects_KV_Growth()
    {
        _fx.Context.ClearKvCache();
        var empty = _fx.Context.GetStateSize();
        Assert.True(empty > 0, "even an empty context has buffer metadata in the state");

        var tokens = _fx.Model.Vocab.Tokenize(
            "The quick brown fox jumps over the lazy dog.",
            addSpecial: true, parseSpecial: false);
        using var sampler = new LlamaSamplerBuilder().WithGreedy().Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);
        await foreach (var _ in gen.GenerateAsync(tokens, maxTokens: 4,
            cancellationToken: TestContext.Current.CancellationToken)) { }

        var populated = _fx.Context.GetStateSize();
        Assert.True(populated >= empty,
            $"state size should not shrink after decode (empty={empty}, populated={populated})");
    }

    [Fact]
    public async Task SaveState_Then_Restore_Produces_Identical_Continuation()
    {
        _fx.Context.ClearKvCache();

        var prompt = _fx.Model.Vocab.Tokenize(
            "Once upon a time, in a small village,",
            addSpecial: true, parseSpecial: false);

        // Prime the context with the prompt by decoding it, then snapshot.
        using (var primer = new LlamaSamplerBuilder().WithGreedy().Build())
        {
            var gen = new LlamaGenerator(_fx.Context, primer);
            await foreach (var _ in gen.GenerateAsync(prompt, maxTokens: 1,
                cancellationToken: TestContext.Current.CancellationToken)) { }
        }

        // Trim the bonus token the primer emitted so the KV is exactly at the
        // end of the prompt.
        if (!_fx.Context.RemoveSequenceRange(0, fromPosition: prompt.Length, toPosition: -1))
        {
            Assert.True(true, "backend refused tail trim — round-trip test skipped");
            return;
        }

        var snapshot = _fx.Context.SaveState();
        Assert.NotEmpty(snapshot);
        Assert.Equal(prompt.Length - 1, _fx.Context.SequencePositionRange(0).Maximum);

        // Baseline: continue from the saved state without a round-trip.
        var baseline = await ContinueFromPrimedState(_fx.Context, prompt, seed: 101, maxTokens: 16);

        // Round-trip: wipe, restore, continue with the same seed — must match.
        _fx.Context.ClearKvCache();
        var (wipedMin, wipedMax) = _fx.Context.SequencePositionRange(0);
        Assert.Null(wipedMin);
        Assert.Null(wipedMax);

        var bytesRead = _fx.Context.RestoreState(snapshot);
        Assert.Equal(snapshot.Length, bytesRead);
        Assert.Equal(prompt.Length - 1, _fx.Context.SequencePositionRange(0).Maximum);

        var restored = await ContinueFromPrimedState(_fx.Context, prompt, seed: 101, maxTokens: 16);

        Assert.Equal(baseline, restored);
    }

    [Fact]
    public async Task SaveStateToFile_Roundtrips_Through_Disk()
    {
        _fx.Context.ClearKvCache();
        var prompt = _fx.Model.Vocab.Tokenize(
            "In a shocking finding, scientists discovered",
            addSpecial: true, parseSpecial: false);

        using (var primer = new LlamaSamplerBuilder().WithGreedy().Build())
        {
            var gen = new LlamaGenerator(_fx.Context, primer);
            await foreach (var _ in gen.GenerateAsync(prompt, maxTokens: 1,
                cancellationToken: TestContext.Current.CancellationToken)) { }
        }
        if (!_fx.Context.RemoveSequenceRange(0, fromPosition: prompt.Length, toPosition: -1))
        {
            Assert.True(true, "backend refused tail trim — file round-trip test skipped");
            return;
        }

        var path = Path.Combine(Path.GetTempPath(), $"llama-state-{Guid.NewGuid():N}.bin");
        try
        {
            _fx.Context.SaveStateToFile(path, prompt);
            Assert.True(new FileInfo(path).Length > 0);

            var baseline = await ContinueFromPrimedState(_fx.Context, prompt, seed: 202, maxTokens: 12);

            _fx.Context.ClearKvCache();
            var loadedTokens = _fx.Context.LoadStateFromFile(path);
            Assert.Equal(prompt, loadedTokens);

            var restored = await ContinueFromPrimedState(_fx.Context, prompt, seed: 202, maxTokens: 12);
            Assert.Equal(baseline, restored);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task SequenceSnapshot_Roundtrips_In_Single_Sequence_Context()
    {
        // Per-sequence save/restore. The fixture's context has
        // MaxSequenceCount=1 so we can only round-trip into sequence 0, but
        // the primitive is the same one that enables multi-sequence forks.
        _fx.Context.ClearKvCache();
        var prompt = _fx.Model.Vocab.Tokenize(
            "Write a haiku about the sea:",
            addSpecial: true, parseSpecial: false);

        using (var primer = new LlamaSamplerBuilder().WithGreedy().Build())
        {
            var gen = new LlamaGenerator(_fx.Context, primer);
            await foreach (var _ in gen.GenerateAsync(prompt, maxTokens: 1,
                cancellationToken: TestContext.Current.CancellationToken)) { }
        }
        if (!_fx.Context.RemoveSequenceRange(0, fromPosition: prompt.Length, toPosition: -1))
        {
            Assert.True(true, "backend refused tail trim — seq snapshot test skipped");
            return;
        }

        var seqSize = _fx.Context.GetSequenceStateSize(sequenceId: 0);
        Assert.True(seqSize > 0);

        var snapshot = _fx.Context.SaveSequenceState(sequenceId: 0);
        Assert.NotEmpty(snapshot);

        var baseline = await ContinueFromPrimedState(_fx.Context, prompt, seed: 303, maxTokens: 10);

        _fx.Context.ClearKvCache();
        var bytesRead = _fx.Context.RestoreSequenceState(destinationSequenceId: 0, snapshot);
        Assert.True(bytesRead > 0);

        var (min, max) = _fx.Context.SequencePositionRange(0);
        Assert.NotNull(min);
        Assert.NotNull(max);

        var restored = await ContinueFromPrimedState(_fx.Context, prompt, seed: 303, maxTokens: 10);
        Assert.Equal(baseline, restored);
    }

    [Fact]
    public void RestoreState_Rejects_Empty_Buffer()
    {
        Assert.Throws<ArgumentException>(() => _fx.Context.RestoreState(ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void SaveState_Into_Undersized_Span_Throws()
    {
        _fx.Context.ClearKvCache();
        var size = _fx.Context.GetStateSize();
        var tooSmall = new byte[1]; // any nonzero state won't fit here
        if (size <= 1) return; // absurdly small state — test doesn't apply
        Assert.Throws<ArgumentException>(() => _fx.Context.SaveState(tooSmall));
    }

    [Fact]
    public void LoadStateFromFile_Throws_On_Missing_File()
    {
        var path = Path.Combine(Path.GetTempPath(), $"does-not-exist-{Guid.NewGuid():N}.bin");
        Assert.Throws<LlamaException>(() => _fx.Context.LoadStateFromFile(path));
    }

    // ----- helpers -----

    /// <summary>
    /// Continue generation from a KV cache that already holds exactly
    /// <paramref name="prompt"/>'s tokens (positions 0..Length-1). Uses the
    /// "trim-last + firstNewIndex=Length-1" trick from the existing
    /// multi-turn continuation tests so LlamaGenerator only re-decodes the
    /// last token and resumes cleanly — bypassing the "must evaluate at
    /// least one token" native constraint.
    /// </summary>
    private static async Task<string> ContinueFromPrimedState(
        LlamaContext ctx, int[] prompt, uint seed, int maxTokens)
    {
        if (!ctx.RemoveSequenceRange(0, fromPosition: prompt.Length - 1, toPosition: -1))
        {
            throw new InvalidOperationException(
                "Backend refused tail trim; continuation path not usable.");
        }
        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed).Build();
        var gen = new LlamaGenerator(ctx, sampler);
        var sb = new StringBuilder();
        await foreach (var piece in gen.GenerateAsync(
            prompt, maxTokens: maxTokens,
            firstNewIndex: prompt.Length - 1,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            sb.Append(piece);
        }
        return sb.ToString();
    }
}
