using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

public class PerformanceTests
{
    // ----- struct layout assertions -----

    [Fact]
    public void PerfContextData_Size_And_Offsets_Match_Pinned()
    {
        Assert.Equal(48, Unsafe.SizeOf<llama_perf_context_data>());
        Assert.Equal(0,  Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.t_start_ms)).ToInt32());
        Assert.Equal(8,  Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.t_load_ms)).ToInt32());
        Assert.Equal(16, Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.t_p_eval_ms)).ToInt32());
        Assert.Equal(24, Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.t_eval_ms)).ToInt32());
        Assert.Equal(32, Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.n_p_eval)).ToInt32());
        Assert.Equal(36, Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.n_eval)).ToInt32());
        Assert.Equal(40, Marshal.OffsetOf<llama_perf_context_data>(nameof(llama_perf_context_data.n_reused)).ToInt32());
    }

    [Fact]
    public void PerfSamplerData_Size_And_Offsets_Match_Pinned()
    {
        Assert.Equal(16, Unsafe.SizeOf<llama_perf_sampler_data>());
        Assert.Equal(0, Marshal.OffsetOf<llama_perf_sampler_data>(nameof(llama_perf_sampler_data.t_sample_ms)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<llama_perf_sampler_data>(nameof(llama_perf_sampler_data.n_sample)).ToInt32());
    }

    // ----- derived-value sanity -----

    [Fact]
    public void PromptTokensPerSecond_Handles_Zero_Time()
    {
        var p = new LlamaContextPerformance(0, 0, 0, 0, 0, 0, 0);
        Assert.Equal(0.0, p.PromptTokensPerSecond);
        Assert.Equal(0.0, p.GeneratedTokensPerSecond);
    }

    [Fact]
    public void TokensPerSecond_Computes_Correctly_For_Known_Inputs()
    {
        var p = new LlamaContextPerformance(
            StartMilliseconds: 0,
            LoadMilliseconds: 0,
            PromptEvalMilliseconds: 500,   // half a second
            TokenEvalMilliseconds: 1000,   // one second
            PromptTokenCount: 50,          // 50 prompt tokens in 0.5s -> 100 tok/s
            GeneratedTokenCount: 40,       // 40 gen tokens in 1s -> 40 tok/s
            GraphReuseCount: 0);
        Assert.Equal(100.0, p.PromptTokensPerSecond, precision: 3);
        Assert.Equal(40.0, p.GeneratedTokensPerSecond, precision: 3);
    }

    [Fact]
    public void SamplerPerformance_SamplesPerSecond_Handles_Zero()
    {
        var s = new LlamaSamplerPerformance(0, 0);
        Assert.Equal(0.0, s.SamplesPerSecond);
    }
}

// Runs against a live context to verify the readback is plumbed correctly.
[Collection(GpuCollection.Name)]
public class PerformanceLiveTests
{
    private readonly GpuGenerationFixture _fx;
    public PerformanceLiveTests(GpuGenerationFixture fx) => _fx = fx;

    [Fact]
    public async Task Context_Perf_Shows_Tokens_After_Generation()
    {
        _fx.Context.ClearKvCache();
        _fx.Context.ResetPerformance();

        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 7).Build();
        sampler.ResetPerformance();

        var gen = new LlamaGenerator(_fx.Context, sampler);
        int piecesEmitted = 0;
        await foreach (var _ in gen.GenerateAsync(
            "Hello.", maxTokens: 10, addSpecial: false, parseSpecial: false))
        {
            piecesEmitted++;
        }

        var ctxPerf = _fx.Context.GetPerformance();
        Assert.True(ctxPerf.PromptTokenCount > 0,
            $"prompt tokens should be recorded, got {ctxPerf.PromptTokenCount}");
        Assert.True(ctxPerf.GeneratedTokenCount > 0,
            $"generated tokens should be recorded, got {ctxPerf.GeneratedTokenCount}");
        Assert.True(ctxPerf.TokenEvalMilliseconds > 0,
            $"eval time should be positive, got {ctxPerf.TokenEvalMilliseconds}");
        Assert.True(ctxPerf.GeneratedTokensPerSecond > 0);

        var samPerf = sampler.GetPerformance();
        Assert.True(samPerf.SampleCount > 0,
            $"sampler should have recorded samples, got {samPerf.SampleCount}");
    }

    [Fact]
    public async Task ResetPerformance_Returns_Counters_To_Native_Floor()
    {
        _fx.Context.ClearKvCache();

        using var sampler = new LlamaSamplerBuilder()
            .WithTemperature(0.7f).WithDistribution(seed: 8).Build();
        var gen = new LlamaGenerator(_fx.Context, sampler);
        await foreach (var _ in gen.GenerateAsync(
            "Warm up.", maxTokens: 5, addSpecial: false, parseSpecial: false)) { }

        var before = _fx.Context.GetPerformance();
        Assert.True(before.GeneratedTokenCount > 1,
            $"expected >1 generated tokens before reset; got {before.GeneratedTokenCount}");

        _fx.Context.ResetPerformance();
        var after = _fx.Context.GetPerformance();
        // llama.cpp's perf_get_data applies max(1, n_eval) / max(1, n_p_eval)
        // so the per-token divisions in its log formatter don't divide by zero.
        // After reset these report as 1, not 0. n_reused has no such floor.
        Assert.Equal(1, after.GeneratedTokenCount);
        Assert.Equal(1, after.PromptTokenCount);
        Assert.Equal(0, after.GraphReuseCount);
        Assert.True(after.GeneratedTokenCount < before.GeneratedTokenCount,
            "reset should have reduced the counters");

        var sBefore = sampler.GetPerformance();
        Assert.True(sBefore.SampleCount > 0,
            $"expected >0 sampler count before reset; got {sBefore.SampleCount}");
        sampler.ResetPerformance();
        // The sampler side doesn't apply the max(1, .) floor — SampleCount
        // really does go to 0.
        Assert.Equal(0, sampler.GetPerformance().SampleCount);
    }
}
