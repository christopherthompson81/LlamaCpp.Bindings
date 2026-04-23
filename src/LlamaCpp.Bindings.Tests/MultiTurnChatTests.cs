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

        // The qualitative recall check needs a model competent enough to
        // actually retrieve the codeword. Sub-3B models often hallucinate
        // here even with perfect KV preservation; the binding-level property
        // (position counter advances) above is the model-agnostic check.
        if (_fx.Capabilities.SkipUnlessMinParameters(3_000_000_000)) return;
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
    public async Task Prefix_Reuse_Produces_Same_Output_As_Full_Rebuild()
    {
        // Correctness contract for ChatSession prefix-cache reuse: given
        // identical prompt tokens and identical sampler seed, running
        // GenerateAsync with firstNewIndex=0 (full rebuild) must produce the
        // same stream as GenerateAsync with firstNewIndex=N when the KV
        // already contains the first N tokens. This is the invariant that
        // lets ChatSession skip re-decoding the common prefix each turn.
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }

        var vocab = _fx.Model.Vocab;
        var tokens = vocab.Tokenize(
            "The rain in Spain falls mainly on the plain, and the hills are alive.",
            addSpecial: false, parseSpecial: false);
        Assert.True(tokens.Length >= 8, "test prompt should be at least 8 tokens");
        int maxGen = 8;

        // Pass A: full rebuild, greedy sampling with fixed seed.
        _fx.Context.ClearKvCache();
        var outputA = new StringBuilder();
        using (var samplerA = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed: 7).Build())
        {
            var genA = new LlamaGenerator(_fx.Context, samplerA);
            await foreach (var p in genA.GenerateAsync(tokens, maxTokens: maxGen,
                                                        firstNewIndex: 0)) outputA.Append(p);
        }

        // Pass B: split decode. Decode tokens[0..half-1] via a throwaway
        // generator, trim the one bonus token it emitted, then run the
        // real generator with firstNewIndex=half against a fresh sampler
        // using the same seed as pass A.
        int half = tokens.Length / 2;
        var firstHalf = new int[half];
        Array.Copy(tokens, firstHalf, half);

        _fx.Context.ClearKvCache();
        using (var throwaway = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed: 9999).Build())
        {
            var genThrow = new LlamaGenerator(_fx.Context, throwaway);
            await foreach (var _ in genThrow.GenerateAsync(firstHalf, maxTokens: 1,
                                                            firstNewIndex: 0)) { }
        }
        // Strip the throwaway's generated token so KV is exactly [0..half-1].
        var trimmed = _fx.Context.RemoveSequenceRange(0, fromPosition: half, toPosition: -1);
        if (!trimmed)
        {
            // Backend refused partial removal — test doesn't apply. (Compact
            // SWA / quantised caches may reject this; UseFullSwaCache default
            // is true so usually succeeds.)
            Assert.True(true, "backend refused partial trim — prefix-reuse test not applicable");
            return;
        }

        var outputB = new StringBuilder();
        using (var samplerB = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed: 7).Build())
        {
            var genB = new LlamaGenerator(_fx.Context, samplerB);
            await foreach (var p in genB.GenerateAsync(tokens, maxTokens: maxGen,
                                                        firstNewIndex: half)) outputB.Append(p);
        }

        Assert.Equal(outputA.ToString(), outputB.ToString());
    }

    [Fact]
    public async Task Continue_After_Partial_Generation_Matches_Single_Pass()
    {
        // Correctness contract for ChatSession's StreamContinuationAsync: the
        // back-off-by-one path on LlamaGenerator (seq_rm the last cached
        // token, re-decode it, continue sampling) must produce the same
        // stream as a single longer generation call, given identical
        // prompt tokens and identical sampler seed. This is the invariant
        // the "Continue" button rides on.
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }

        var vocab = _fx.Model.Vocab;
        var prompt = vocab.Tokenize(
            "List three colors:", addSpecial: false, parseSpecial: false);
        int totalGen = 16;
        int firstHalf = 6;

        // Pass A: one shot, generate the full thing.
        _fx.Context.ClearKvCache();
        var outputA = new StringBuilder();
        using (var samplerA = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed: 13).Build())
        {
            var genA = new LlamaGenerator(_fx.Context, samplerA);
            await foreach (var p in genA.GenerateAsync(prompt, maxTokens: totalGen)) outputA.Append(p);
        }

        // Pass B: generate first half, then "continue" via back-off-by-one.
        // Track all tokens that land in the KV cache via onTokenDecoded so we
        // can pass them as the prompt for the continuation.
        _fx.Context.ClearKvCache();
        var decodedSoFar = new List<int>(prompt);
        var outputB = new StringBuilder();
        using (var samplerB1 = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed: 13).Build())
        {
            var genB1 = new LlamaGenerator(_fx.Context, samplerB1);
            await foreach (var p in genB1.GenerateAsync(
                prompt, maxTokens: firstHalf, onTokenDecoded: decodedSoFar.Add)) outputB.Append(p);
        }

        // Now do the continuation — back off one, fresh sampler same seed.
        var trimmed = _fx.Context.RemoveSequenceRange(
            0, fromPosition: decodedSoFar.Count - 1, toPosition: -1);
        if (!trimmed)
        {
            Assert.True(true, "backend refused partial trim — continuation path not applicable");
            return;
        }

        using (var samplerB2 = new LlamaSamplerBuilder()
            .WithTemperature(0.0f).WithDistribution(seed: 13).Build())
        {
            var genB2 = new LlamaGenerator(_fx.Context, samplerB2);
            await foreach (var p in genB2.GenerateAsync(
                decodedSoFar,
                maxTokens: totalGen - firstHalf,
                firstNewIndex: decodedSoFar.Count - 1,
                onTokenDecoded: decodedSoFar.Add)) outputB.Append(p);
        }

        Assert.Equal(outputA.ToString(), outputB.ToString());
    }

    [Fact]
    public async Task Decoding_After_Tail_Trim_Resumes_From_Trim_Position()
    {
        // This is the invariant prefix-cache reuse rides on: after a
        // seq_rm(seq, K, -1) that succeeds, the next llama_batch_get_one
        // decode should position its tokens starting at K, not at the
        // pre-trim length. If this test ever fails we have to switch the
        // ChatSession prefix-reuse path to explicit-position batches.
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 41).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);

        // Step 1: decode a prompt + ~10 tokens. Captures (min=0, max=some N).
        await foreach (var _ in gen.GenerateAsync("Count: one two three", maxTokens: 10,
                                                   addSpecial: false, parseSpecial: false)) { }
        var (_, maxBefore) = _fx.Context.SequencePositionRange(0);
        Assert.NotNull(maxBefore);
        int lenBefore = maxBefore!.Value + 1; // positions are 0-indexed inclusive

        // Step 2: trim to half. Skip rest of test if backend refused.
        int trimAt = lenBefore / 2;
        if (!_fx.Context.RemoveSequenceRange(0, fromPosition: trimAt, toPosition: -1))
        {
            Assert.True(true, "backend refused mid-sequence trim — test does not apply");
            return;
        }
        var (_, maxAfterTrim) = _fx.Context.SequencePositionRange(0);
        Assert.Equal(trimAt - 1, maxAfterTrim);

        // Step 3: decode one more piece through LlamaGenerator. If auto-
        // positioning after seq_rm is correct, positions should extend from
        // trimAt onward; the new max should be > trimAt-1 but <= trimAt + new.
        int emittedSecond = 0;
        await foreach (var _ in gen.GenerateAsync(" four", maxTokens: 5,
                                                   addSpecial: false, parseSpecial: false))
        {
            emittedSecond++;
        }
        var (_, maxAfterResume) = _fx.Context.SequencePositionRange(0);
        Assert.NotNull(maxAfterResume);
        Assert.True(maxAfterResume > maxAfterTrim,
            $"Decoding after seq_rm should extend positions from the trim point. " +
            $"Got max={maxAfterResume}, expected > {maxAfterTrim}.");
        // The new max must be close to trimAt + (new prompt tokens + generated) —
        // if it landed near lenBefore + new, that's the bug we're guarding
        // against (auto-positioning didn't reset).
        Assert.True(maxAfterResume < lenBefore + 50,
            $"Suspicious: decode resumed at an implausibly high position. " +
            $"max={maxAfterResume}, lenBefore={lenBefore}. Possibly auto-positioning " +
            $"didn't reset after seq_rm.");
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
