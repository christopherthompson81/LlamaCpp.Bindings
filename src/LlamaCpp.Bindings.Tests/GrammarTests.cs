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

[Collection(GpuCollection.Name)]
public class GrammarGenerationTests
{
    private readonly GpuGenerationFixture _fx;
    public GrammarGenerationTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public async Task Simple_Grammar_Constrains_Output_To_Literal_Choices()
    {
        if (_fx.Context is null || _fx.Model is null) { _fx.SkipMessage(); return; }
        _fx.Context.ClearKvCache();

        // Minimal grammar: the model must emit either "yes" or "no". Once
        // the grammar is satisfied, the generator's grammar-termination gate
        // (backed by LlamaSampler.IsGrammarSatisfied) exits cleanly before
        // the next sample/accept cycle would throw.
        var yesNo = new LlamaGrammar("root ::= \"yes\" | \"no\"");

        using var sampler = new LlamaSamplerBuilder()
            .WithGrammar(_fx.Model.Vocab, yesNo)
            .WithGreedy()
            .Build();
        Assert.True(sampler.HasGrammar);

        var gen = new LlamaGenerator(_fx.Context, sampler);
        var sb = new System.Text.StringBuilder();
        await foreach (var p in gen.GenerateAsync(
            "Is the sky blue? Answer yes or no.",
            maxTokens: 20, addSpecial: false, parseSpecial: false))
        {
            sb.Append(p);
        }
        var output = sb.ToString().Trim();
        Assert.True(output == "yes" || output == "no",
            $"grammar should constrain output to 'yes' or 'no'; got \"{output}\"");
    }

    // NOTE: no JSON-mode generation test.
    //
    // The simple literal-choice grammar (yes/no) works end-to-end with the
    // grammar-termination detection we just added. The bundled JSON grammar
    // has trailing-ws productions that let the grammar accept arbitrary
    // whitespace after a closed object — my IsGrammarSatisfied check
    // correctly reports "still permits whitespace," so the generator keeps
    // going, and at some point the native grammar engine hits an internal
    // "empty stack" state not captured by the apply()-probe approach and
    // throws.
    //
    // The remaining fix is probably to bind llama_grammar_free-style
    // lifecycle helpers or to port more of the CLI's grammar-complete
    // heuristics. Tracked as a follow-up in #10's comments; for now,
    // complex grammars with trailing-ws productions should bound generation
    // via maxTokens, not rely on grammar termination.

    [Fact]
    public void IsGrammarSatisfied_Is_False_On_Non_Grammar_Sampler()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        using var plain = new LlamaSamplerBuilder().WithGreedy().Build();
        Assert.False(plain.HasGrammar);
        Assert.False(plain.IsGrammarSatisfied(_fx.Model.Vocab));
    }

    [Fact]
    public void IsGrammarSatisfied_Is_False_Before_Any_Accept()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        // Fresh grammar sampler hasn't advanced state; there are still valid
        // non-EOG tokens to emit.
        using var s = new LlamaSamplerBuilder()
            .WithGrammar(_fx.Model.Vocab, new LlamaGrammar("root ::= \"yes\" | \"no\""))
            .WithGreedy()
            .Build();
        Assert.True(s.HasGrammar);
        Assert.False(s.IsGrammarSatisfied(_fx.Model.Vocab),
            "grammar allows 'yes' or 'no' before any token is accepted");
    }
}
