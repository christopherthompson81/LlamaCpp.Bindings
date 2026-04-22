namespace LlamaCpp.Bindings;

/// <summary>
/// Snapshot of a context's timing and throughput stats. Retrieved via
/// <see cref="LlamaContext.GetPerformance"/>.
/// </summary>
/// <param name="StartMilliseconds">
/// Absolute start time (llama.cpp wall-clock, not process time). Compare
/// against itself between runs; absolute value isn't meaningful.
/// </param>
/// <param name="LoadMilliseconds">Wall time spent loading the model.</param>
/// <param name="PromptEvalMilliseconds">Time spent evaluating the prompt (prefill).</param>
/// <param name="TokenEvalMilliseconds">Time spent generating tokens (decode).</param>
/// <param name="PromptTokenCount">
/// How many prompt tokens were evaluated. Note: llama.cpp clamps this to a
/// minimum of 1 internally (so the per-token-time division in its log
/// formatter doesn't divide by zero). If you just called
/// <see cref="LlamaContext.ResetPerformance"/>, expect the value to read
/// back as <c>1</c>, not <c>0</c>.
/// </param>
/// <param name="GeneratedTokenCount">
/// How many tokens have been generated. Also clamped to a minimum of 1 by
/// the native layer.
/// </param>
/// <param name="GraphReuseCount">
/// How many times an internal ggml compute graph was reused — proxy for cache
/// efficiency across decode steps.
/// </param>
public readonly record struct LlamaContextPerformance(
    double StartMilliseconds,
    double LoadMilliseconds,
    double PromptEvalMilliseconds,
    double TokenEvalMilliseconds,
    int PromptTokenCount,
    int GeneratedTokenCount,
    int GraphReuseCount)
{
    /// <summary>Average tokens/sec across prompt evaluation. 0 if no prompt has been run.</summary>
    public double PromptTokensPerSecond =>
        PromptEvalMilliseconds > 0 ? PromptTokenCount * 1000.0 / PromptEvalMilliseconds : 0.0;

    /// <summary>Average tokens/sec across generation. 0 if no generation has run.</summary>
    public double GeneratedTokensPerSecond =>
        TokenEvalMilliseconds > 0 ? GeneratedTokenCount * 1000.0 / TokenEvalMilliseconds : 0.0;
}

/// <summary>
/// Snapshot of a sampler chain's timing and throughput stats. Retrieved via
/// <see cref="LlamaSampler.GetPerformance"/>.
/// </summary>
/// <param name="SampleMilliseconds">Total wall time spent inside sampler stages.</param>
/// <param name="SampleCount">Number of sampling events that have occurred.</param>
public readonly record struct LlamaSamplerPerformance(
    double SampleMilliseconds,
    int SampleCount)
{
    /// <summary>Average samples/sec. 0 if no sampling has happened.</summary>
    public double SamplesPerSecond =>
        SampleMilliseconds > 0 ? SampleCount * 1000.0 / SampleMilliseconds : 0.0;
}
