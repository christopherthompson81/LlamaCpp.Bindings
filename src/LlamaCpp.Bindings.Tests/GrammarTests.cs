namespace LlamaCpp.Bindings.Tests;

public class GrammarBuilderTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public GrammarBuilderTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Grammar_Adds_Non_Terminal_Stage()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }

        using var s = new LlamaSamplerBuilder()
            .WithGrammar(_fx.Model.Vocab, LlamaGrammar.Json)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 1)
            .Build();
        Assert.Equal(3, s.ChainLength);
    }

    [Fact]
    public void Grammar_Rejects_Empty_Source()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        Assert.Throws<ArgumentException>(() =>
            new LlamaSamplerBuilder()
                .WithGrammar(_fx.Model.Vocab, new LlamaGrammar(""))
                .WithGreedy()
                .Build());
    }

    // NOTE: We do NOT test "invalid grammar throws LlamaException" because
    // llama.cpp's grammar parser aborts the process on syntax errors rather
    // than returning NULL. Testing this would crash the test host. The
    // public API documents this caveat; callers are expected to validate
    // their GBNF against llama.cpp's CLI before handing it to the binding.

    [Fact]
    public void LazyGrammar_Accepts_Patterns_And_Tokens()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        using var s = new LlamaSamplerBuilder()
            .WithLazyGrammar(_fx.Model.Vocab, new LlamaLazyGrammar(
                Grammar: LlamaGrammar.Json,
                TriggerPatterns: new[] { "^```json" },
                TriggerTokens: Array.Empty<int>()))
            .WithGreedy()
            .Build();
        Assert.Equal(2, s.ChainLength);
    }
}

// NOTE: no generation tests here.
//
// llama.cpp's grammar sampler throws an uncatchable C++ runtime_error
// ("Unexpected empty grammar stack after accepting piece") the moment the
// grammar is fully satisfied and another Accept is attempted — even if that
// Accept would have been an EOG. Safely driving grammar-constrained
// generation to completion requires a managed-side state machine that
// detects grammar-done BEFORE the next Sample/Accept, which in turn
// requires binding llama_sampler_apply (currently in the Tier-2 "custom
// sampling" category, not yet exposed).
//
// The builder tests above cover the chain-composition contract. The
// grammar-generation story is filed as a separate GH issue.
