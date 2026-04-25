using System.Text;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Multi-session (parallel seq_id) tests for issue #5. A single context with
/// <see cref="LlamaContextParameters.MaxSequenceCount"/> &gt; 1 should be
/// able to drive several isolated conversations, each pinned to its own
/// sequence slot, with KV state that doesn't leak between them.
///
/// Uses its own fixture because the shared <see cref="GpuGenerationFixture"/>
/// pins <c>MaxSequenceCount = 1</c>.
/// </summary>
public class LlamaSessionTests : IClassFixture<MultiSeqFixture>
{
    private readonly MultiSeqFixture _fx;
    public LlamaSessionTests(MultiSeqFixture fx) => _fx = fx;

    // ----- Allocation / lifecycle -----

    [Fact]
    public void CreateSession_Returns_Distinct_Sequence_Ids()
    {
        using var a = _fx.Context.CreateSession();
        using var b = _fx.Context.CreateSession();
        using var c = _fx.Context.CreateSession();

        Assert.NotEqual(a.SequenceId, b.SequenceId);
        Assert.NotEqual(b.SequenceId, c.SequenceId);
        Assert.NotEqual(a.SequenceId, c.SequenceId);
        Assert.All(new[] { a.SequenceId, b.SequenceId, c.SequenceId },
            id => Assert.InRange(id, 0, _fx.Context.MaxSequenceCount - 1));
    }

    [Fact]
    public void CreateSession_Throws_When_Pool_Exhausted()
    {
        var held = new List<LlamaSession>();
        try
        {
            for (int i = 0; i < _fx.Context.MaxSequenceCount; i++)
            {
                held.Add(_fx.Context.CreateSession());
            }
            Assert.Throws<InvalidOperationException>(() => _fx.Context.CreateSession());
        }
        finally
        {
            foreach (var s in held) s.Dispose();
        }
    }

    [Fact]
    public void Disposing_Session_Returns_Slot_To_Pool()
    {
        int taken;
        using (var s = _fx.Context.CreateSession())
        {
            taken = s.SequenceId;
        }
        // The just-released slot should be reusable without growing the pool.
        using var reused = _fx.Context.CreateSession();
        // Reuse happens to be first-fit lowest free id, so we expect the same
        // id back — belt and suspenders, a strict match also validates the
        // pool's bookkeeping.
        Assert.Equal(taken, reused.SequenceId);
    }

    [Fact]
    public void Session_PositionRange_Is_Empty_Before_Decode()
    {
        using var s = _fx.Context.CreateSession();
        var (min, max) = s.PositionRange;
        Assert.Null(min);
        Assert.Null(max);
    }

    // ----- Generation isolation -----

    [Fact]
    public async Task Session_Generations_Do_Not_Leak_Into_Each_Other()
    {
        // Drive two sessions against the same context with completely
        // different prompts. After both finish, each session's KV range
        // should reflect its own prompt + emitted tokens, not the other's.
        using var sessionA = _fx.Context.CreateSession();
        using var sessionB = _fx.Context.CreateSession();
        using var samplerA = new LlamaSamplerBuilder().WithGreedy().Build();
        using var samplerB = new LlamaSamplerBuilder().WithGreedy().Build();
        var genA = new LlamaGenerator(sessionA, samplerA);
        var genB = new LlamaGenerator(sessionB, samplerB);

        var outA = new StringBuilder();
        await foreach (var p in genA.GenerateAsync("Once upon a time", maxTokens: 8,
            addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            outA.Append(p);
        }

        var outB = new StringBuilder();
        await foreach (var p in genB.GenerateAsync("The answer to 1 + 1 is", maxTokens: 8,
            addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            outB.Append(p);
        }

        Assert.False(string.IsNullOrWhiteSpace(outA.ToString()));
        Assert.False(string.IsNullOrWhiteSpace(outB.ToString()));
        Assert.NotEqual(outA.ToString(), outB.ToString());

        // Both sessions must have their own non-empty position ranges;
        // neither should have accumulated the other's tokens.
        var rangeA = sessionA.PositionRange;
        var rangeB = sessionB.PositionRange;
        Assert.NotNull(rangeA.Maximum);
        Assert.NotNull(rangeB.Maximum);
        Assert.True(rangeA.Maximum >= 4, "session A KV did not advance");
        Assert.True(rangeB.Maximum >= 4, "session B KV did not advance");
    }

    [Fact]
    public async Task Session_Matches_Plain_Single_Context_Output()
    {
        // Semantic: generating on seq_id=0 through the implicit legacy ctor
        // should match generating on a CreateSession()'d session (which
        // allocates id=0 first). Smoke-tests that the session-aware batch
        // builder produces equivalent decodes to llama_batch_get_one on the
        // default slot.
        var prompt = "The capital of France is";

        using var solo = new LlamaContext(_fx.Model, new LlamaContextParameters
        {
            ContextSize = 512,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
            OffloadKQV = false, // use CPU-stable path to avoid batch-shape FP drift
        });
        using var soloSampler = new LlamaSamplerBuilder().WithGreedy().Build();
        var soloGen = new LlamaGenerator(solo, soloSampler);
        var soloOut = new StringBuilder();
        await foreach (var p in soloGen.GenerateAsync(prompt, maxTokens: 8,
            addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            soloOut.Append(p);
        }

        // Build a second stand-alone context on CPU explicitly so both legs
        // run on the same backend. The fixture's context is multi-seq; we
        // don't reuse it here to keep the comparison apples-to-apples.
        using var pair = new LlamaContext(_fx.Model, new LlamaContextParameters
        {
            ContextSize = 512,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 4,
            OffloadKQV = false,
        });
        using var session = pair.CreateSession();
        using var sampler = new LlamaSamplerBuilder().WithGreedy().Build();
        var gen = new LlamaGenerator(session, sampler);
        var sessionOut = new StringBuilder();
        await foreach (var p in gen.GenerateAsync(prompt, maxTokens: 8,
            addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            sessionOut.Append(p);
        }

        Assert.Equal(soloOut.ToString(), sessionOut.ToString());
    }

    [Fact]
    public async Task Concurrent_Sessions_Produce_Correct_Isolated_Output()
    {
        // Drive two sessions in parallel Tasks. Each should complete with
        // output derivable only from its own prompt — the decode lock inside
        // LlamaContext ensures decode+sample pairs are serialized so the
        // shared logits buffer is never read across-sessions.
        using var sessionA = _fx.Context.CreateSession();
        using var sessionB = _fx.Context.CreateSession();

        async Task<string> Run(LlamaSession s, string prompt)
        {
            using var sampler = new LlamaSamplerBuilder().WithGreedy().Build();
            var gen = new LlamaGenerator(s, sampler);
            var sb = new StringBuilder();
            await foreach (var p in gen.GenerateAsync(prompt, maxTokens: 8,
                addSpecial: false, parseSpecial: false,
                cancellationToken: TestContext.Current.CancellationToken))
            {
                sb.Append(p);
            }
            return sb.ToString();
        }

        var taskA = Task.Run(() => Run(sessionA, "Once upon a time"));
        var taskB = Task.Run(() => Run(sessionB, "The answer to 1 + 1 is"));
        await Task.WhenAll(taskA, taskB);

        var outA = await taskA;
        var outB = await taskB;

        Assert.False(string.IsNullOrWhiteSpace(outA));
        Assert.False(string.IsNullOrWhiteSpace(outB));
        Assert.NotEqual(outA, outB);

        // Each session's KV should hold only its own conversation. Maximum
        // position is "prompt_len + emitted - 1" roughly; enforcing a hard
        // upper bound catches the worst case where both conversations
        // ended up in one sequence.
        var (_, maxA) = sessionA.PositionRange;
        var (_, maxB) = sessionB.PositionRange;
        Assert.NotNull(maxA);
        Assert.NotNull(maxB);
        Assert.True(maxA < 64, $"session A KV grew unexpectedly large: max={maxA}");
        Assert.True(maxB < 64, $"session B KV grew unexpectedly large: max={maxB}");
    }

    [Fact]
    public async Task ClearHistory_Resets_Session_Range_Without_Touching_Other_Sessions()
    {
        using var a = _fx.Context.CreateSession();
        using var b = _fx.Context.CreateSession();
        using var sA = new LlamaSamplerBuilder().WithGreedy().Build();
        using var sB = new LlamaSamplerBuilder().WithGreedy().Build();
        var genA = new LlamaGenerator(a, sA);
        var genB = new LlamaGenerator(b, sB);

        await foreach (var _ in genA.GenerateAsync("Hello world", maxTokens: 3,
            addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken)) { }
        await foreach (var _ in genB.GenerateAsync("Goodbye world", maxTokens: 3,
            addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken)) { }

        var aMaxBefore = a.PositionRange.Maximum;
        var bMaxBefore = b.PositionRange.Maximum;
        Assert.NotNull(aMaxBefore);
        Assert.NotNull(bMaxBefore);

        a.ClearHistory();

        Assert.Null(a.PositionRange.Maximum);
        Assert.Equal(bMaxBefore, b.PositionRange.Maximum);
    }
}

/// <summary>
/// Fixture for multi-session tests — a context with room for several
/// sequence slots. CPU-only so byte-equivalence between solo and session
/// decode is actually assertable (the GPU split-decode kernel path
/// diverges; see docs/prefix_reuse_investigation.md).
/// </summary>
public sealed class MultiSeqFixture : IDisposable
{
    public LlamaModel Model { get; }
    public LlamaContext Context { get; }

    public MultiSeqFixture()
    {
        LlamaBackend.Initialize();
        var path = TestModelProvider.EnsureModelPath();
        Model = new LlamaModel(path, new LlamaModelParameters
        {
            GpuLayerCount = 0, // CPU — deterministic, avoids the GPU batched-attn FP drift
            UseMmap = true,
        });
        Context = new LlamaContext(Model, new LlamaContextParameters
        {
            ContextSize = 512,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 4,
            OffloadKQV = false,
        });
    }

    public void Dispose()
    {
        Context.Dispose();
        Model.Dispose();
    }
}
