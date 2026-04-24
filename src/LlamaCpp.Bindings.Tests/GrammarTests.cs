namespace LlamaCpp.Bindings.Tests;

public class GrammarBuilderTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public GrammarBuilderTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Grammar_Is_Held_Separately_From_The_Chain()
    {

        using var s = new LlamaSamplerBuilder()
            .WithGrammar(_fx.Model.Vocab, LlamaGrammar.Json)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 1)
            .Build();
        // Grammar is NOT in the chain (rejection-sampling design).
        // Chain stages: temperature + distribution. Grammar is owned
        // separately and runs as a post-pick validator.
        Assert.Equal(2, s.ChainLength);
        Assert.True(s.HasGrammar);
    }

    [Fact]
    public void Grammar_Rejects_Empty_Source()
    {
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
        using var s = new LlamaSamplerBuilder()
            .WithLazyGrammar(_fx.Model.Vocab, new LlamaLazyGrammar(
                Grammar: LlamaGrammar.Json,
                TriggerPatterns: new[] { "^```json" },
                TriggerTokens: Array.Empty<int>()))
            .WithGreedy()
            .Build();
        // Grammar held separately; chain only has greedy.
        Assert.Equal(1, s.ChainLength);
        Assert.True(s.HasGrammar);
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

    [Fact]
    public async Task Json_Grammar_Produces_Parseable_Json()
    {
        _fx.Context.ClearKvCache();

        // With the rejection-sampling design, grammar is held separately from
        // the chain. This avoids the reject/accept disagreement in llama.cpp's
        // grammar engine that crashed the host on complex grammars in the
        // previous in-chain design.
        using var sampler = new LlamaSamplerBuilder()
            .WithTopK(40)
            .WithTemperature(0.7f)
            .WithGrammar(_fx.Model.Vocab, LlamaGrammar.Json)
            .WithDistribution(seed: 123)
            .Build();

        var gen = new LlamaGenerator(_fx.Context, sampler);
        var sb = new System.Text.StringBuilder();
        await foreach (var p in gen.GenerateAsync(
            "Emit a JSON object with a name and age. Output JSON only.\n",
            maxTokens: 200, addSpecial: false, parseSpecial: false))
        {
            sb.Append(p);
        }
        var output = sb.ToString().Trim();
        Assert.False(string.IsNullOrWhiteSpace(output),
            "grammar-constrained generation produced empty output");

        using var doc = System.Text.Json.JsonDocument.Parse(output);
        Assert.NotNull(doc);
    }

    [Fact]
    public void IsGrammarSatisfied_Is_False_On_Non_Grammar_Sampler()
    {
        using var plain = new LlamaSamplerBuilder().WithGreedy().Build();
        Assert.False(plain.HasGrammar);
        Assert.False(plain.IsGrammarSatisfied(_fx.Model.Vocab));
    }

    [Fact]
    public void IsGrammarSatisfied_Is_False_Before_Any_Accept()
    {
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
