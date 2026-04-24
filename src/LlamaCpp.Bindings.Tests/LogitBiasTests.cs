using System.Runtime.InteropServices;
using System.Text;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

public class LogitBiasTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public LogitBiasTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Struct_Layout_Matches_Pinned()
    {
        Assert.Equal(8, System.Runtime.CompilerServices.Unsafe.SizeOf<llama_logit_bias>());
        Assert.Equal(0, Marshal.OffsetOf<llama_logit_bias>(nameof(llama_logit_bias.token)).ToInt32());
        Assert.Equal(4, Marshal.OffsetOf<llama_logit_bias>(nameof(llama_logit_bias.bias)).ToInt32());
    }

    [Fact]
    public void Empty_Bias_List_Is_Noop()
    {
        using var s = new LlamaSamplerBuilder()
            .WithLogitBias(_fx.Model.Vocab, Array.Empty<(int, float)>())
            .WithGreedy()
            .Build();
        // Greedy + empty bias = just greedy.
        Assert.Equal(1, s.ChainLength);
    }

    [Fact]
    public void Bias_With_Valid_Tokens_Adds_Stage()
    {
        var v = _fx.Model.Vocab;
        using var s = new LlamaSamplerBuilder()
            .WithLogitBias(v, new[] { (0, 1.0f), (1, -2.0f) })
            .WithTemperature(0.7f)
            .WithDistribution(seed: 1)
            .Build();
        Assert.Equal(3, s.ChainLength);
    }

    [Fact]
    public void Bias_Rejects_Out_Of_Range_Token()
    {
        var v = _fx.Model.Vocab;
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlamaSamplerBuilder()
                .WithLogitBias(v, new[] { (v.TokenCount, 0.0f) })
                .WithGreedy()
                .Build());

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlamaSamplerBuilder()
                .WithLogitBias(v, new[] { (-1, 0.0f) })
                .WithGreedy()
                .Build());
    }

    [Fact]
    public void BannedTokens_Applies_NegativeInfinity()
    {
        var v = _fx.Model.Vocab;
        using var s = new LlamaSamplerBuilder()
            .WithBannedTokens(v, new[] { 0, 1, 2 })
            .WithGreedy()
            .Build();
        Assert.Equal(2, s.ChainLength); // logit-bias + greedy
    }
}

[Collection(GpuCollection.Name)]
public class LogitBiasBehaviouralTests
{
    private readonly GpuGenerationFixture _fx;
    public LogitBiasBehaviouralTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public async Task Banning_Specific_Tokens_Suppresses_Their_Output()
    {
        _fx.Context.ClearKvCache();

        var v = _fx.Model.Vocab;

        // Tokenize the word "hello" and ban whatever tokens make it up. Then
        // prompt for "hello" explicitly; the model should route around.
        var helloTokens = v.Tokenize("hello", addSpecial: false, parseSpecial: false);
        Assert.NotEmpty(helloTokens);

        using var sampler = new LlamaSamplerBuilder()
            .WithBannedTokens(v, helloTokens)
            .WithTemperature(0.7f)
            .WithDistribution(seed: 42)
            .Build();

        var gen = new LlamaGenerator(_fx.Context, sampler);
        var sb = new StringBuilder();
        await foreach (var p in gen.GenerateAsync(
            "User: Please say the single word hello.\nAssistant: ",
            maxTokens: 30, addSpecial: false, parseSpecial: false,
            cancellationToken: TestContext.Current.CancellationToken))
        {
            sb.Append(p);
        }
        var output = sb.ToString();
        Assert.False(string.IsNullOrWhiteSpace(output));

        // None of the banned tokens should appear in the output. Re-tokenize
        // and check no banned id is present.
        var outputTokens = v.Tokenize(output, addSpecial: false, parseSpecial: false);
        var banned = new HashSet<int>(helloTokens);
        foreach (var t in outputTokens)
        {
            Assert.DoesNotContain(t, banned);
        }
    }
}
