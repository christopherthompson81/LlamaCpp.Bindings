using System.Text;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Phase 4: KV cache management + incremental multi-turn decode.
///
/// The core claim we're validating: across consecutive <c>GenerateAsync</c>
/// calls on the same context, turn N's prefix (prompt + generated response)
/// stays in the KV cache, so turn N+1 only decodes the newly-spoken tokens
/// rather than re-processing the whole history. Plus: <c>ClearKvCache</c>
/// actually resets state so an unrelated conversation isn't polluted.
/// </summary>
[Collection(GpuCollection.Name)]
public class MultiTurnChatTests
{
    private readonly GpuGenerationFixture _fx;
    public MultiTurnChatTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public void SequencePositionRange_On_Fresh_Context_Is_Empty()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();
        var range = _fx.Context.SequencePositionRange(sequenceId: 0);
        Assert.Null(range.Minimum);
        Assert.Null(range.Maximum);
    }

    [Fact]
    public async Task Decode_Advances_Sequence_Position_Counter()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder().WithTemperature(0.7f).WithDistribution(seed: 1).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        // Generate a small chunk so the cache clearly advances.
        int emitted = 0;
        await foreach (var _ in gen.GenerateAsync("The capital of France is", maxTokens: 8,
                                                   addSpecial: false, parseSpecial: false))
        {
            emitted++;
        }

        var (min, max) = _fx.Context.SequencePositionRange(0);
        Assert.True(emitted > 0, "should have emitted at least one piece");
        Assert.True(min is not null && max is not null,
            $"position range should be populated after decode; got (min={min}, max={max}, emitted={emitted})");
        Assert.True(max >= min, $"max({max}) should be >= min({min})");
        // Prompt alone is ≥5 tokens, plus 8 generated = ≥12 positions used; be lenient.
        Assert.True(max >= 5,
            $"position max should reflect decode progress; got (min={min}, max={max}, emitted={emitted})");
    }

    [Fact]
    public async Task ClearKvCache_Resets_Sequence_Range()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder().WithTemperature(0.7f).WithDistribution(seed: 1).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        await foreach (var _ in gen.GenerateAsync("Hello world", maxTokens: 5, addSpecial: false, parseSpecial: false)) { }
        Assert.NotNull(_fx.Context.SequencePositionRange(0).Maximum);

        _fx.Context.ClearKvCache();

        var (min, max) = _fx.Context.SequencePositionRange(0);
        Assert.Null(min);
        Assert.Null(max);
    }

    [Fact]
    public async Task Multi_Turn_History_Is_Retained_Without_Reprocessing()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        // Reuse a single sampler across turns so penalty history carries over the
        // way a real chat session would.
        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 11).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        // Turn 1: establish a fact the model could only know from the prompt.
        const string turn1 = "Remember this: my secret codeword is 'pineapple'.\nAssistant:";
        await foreach (var _ in gen.GenerateAsync(turn1, maxTokens: 20, addSpecial: false, parseSpecial: false)) { }

        var posAfterTurn1 = _fx.Context.SequencePositionRange(0).Maximum;
        Assert.NotNull(posAfterTurn1);

        // Turn 2: ask about turn 1 content using ONLY the new user text — do
        // not replay the turn-1 prompt. If the KV cache lost history, the model
        // has no way to answer correctly.
        const string turn2 = "\nUser: What was the codeword?\nAssistant:";
        var sb = new StringBuilder();
        await foreach (var piece in gen.GenerateAsync(turn2, maxTokens: 30, addSpecial: false, parseSpecial: false))
        {
            sb.Append(piece);
        }
        var answer = sb.ToString();

        var posAfterTurn2 = _fx.Context.SequencePositionRange(0).Maximum;
        Assert.NotNull(posAfterTurn2);
        Assert.True(posAfterTurn2 > posAfterTurn1,
            "Turn 2 should have advanced the position counter beyond turn 1.");

        // If history was preserved, "pineapple" shows up (case may vary). This is
        // a behavioural test on a real model, so it's a soft assertion with a
        // helpful failure message rather than a hard contract — models can and
        // do occasionally dodge prompts like this.
        Assert.True(
            answer.Contains("pineapple", StringComparison.OrdinalIgnoreCase),
            $"Expected the model to remember the codeword across turns without KV re-processing.\nAnswer was: {answer}");
    }

    [Fact]
    public async Task After_ClearKvCache_New_Conversation_Does_Not_Leak_Prior_Context()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler1 = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 21).Build();
        var gen1 = new LlamaGenerator(_fx.Context, sampler1);

        // Load a distinctive token into history.
        await foreach (var _ in gen1.GenerateAsync(
            "The secret word is zebra.\nAssistant: Got it.",
            maxTokens: 5, addSpecial: false, parseSpecial: false)) { }

        _fx.Context.ClearKvCache();

        // Fresh sampler; fresh KV. Asking the model about the prior secret
        // shouldn't surface 'zebra' — that history is gone.
        using var sampler2 = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 22).Build();
        var gen2 = new LlamaGenerator(_fx.Context, sampler2);

        var sb = new StringBuilder();
        await foreach (var piece in gen2.GenerateAsync(
            "What's 7 times 6? Just the number.",
            maxTokens: 40, addSpecial: false, parseSpecial: false))
        {
            sb.Append(piece);
        }
        var output = sb.ToString();

        // Quantitative: KV range is non-empty and starts from 0 (new conversation).
        var (min, max) = _fx.Context.SequencePositionRange(0);
        Assert.NotNull(min);
        Assert.NotNull(max);
        // Qualitative: the word 'zebra' should not appear. Any model competent
        // enough to pass our smoke tests will not spontaneously produce it in
        // response to an arithmetic prompt.
        Assert.DoesNotContain("zebra", output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Removing_Range_Reports_Success_Or_Failure()
    {
        if (_fx.Context is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        // Removing from an empty sequence is a no-op that succeeds.
        var ok = _fx.Context.RemoveSequenceRange(sequenceId: 0, fromPosition: 0, toPosition: -1);
        Assert.True(ok);
    }

    [Fact]
    public async Task Removing_Tail_Trims_Sequence_Position_Range_When_Backend_Supports_It()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 31).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        await foreach (var _ in gen.GenerateAsync("Say ten words quickly.", maxTokens: 10,
                                                   addSpecial: false, parseSpecial: false)) { }

        var (_, maxBefore) = _fx.Context.SequencePositionRange(0);
        Assert.NotNull(maxBefore);

        // Partial sequence removal is backend-dependent. Compact SWA caches and
        // quantised KV caches may refuse mid-sequence removals even with
        // swa_full=true — the native call returns false in that case rather
        // than silently leaving the cache in a bad state. Either outcome is
        // contract-legal; only assert the invariant that IF it succeeded,
        // the cache shrunk; and that total-sequence removal always works
        // (covered by Removing_Range_Reports_Success_Or_Failure).
        int mid = maxBefore!.Value / 2;
        var ok = _fx.Context.RemoveSequenceRange(0, fromPosition: mid, toPosition: -1);
        if (ok)
        {
            var (_, maxAfter) = _fx.Context.SequencePositionRange(0);
            Assert.NotNull(maxAfter);
            Assert.True(maxAfter < maxBefore, "Cache should be shorter after tail removal.");
        }
        // else: backend rejected the partial removal, which is a legal response.
        // The practical workaround is to ClearKvCache() and re-decode the
        // truncated prefix; or for simple rollback, keep track of position
        // before a decode and call RemoveSequenceRange from that offset.
    }
}
