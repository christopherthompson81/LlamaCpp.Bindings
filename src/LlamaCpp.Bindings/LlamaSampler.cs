using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;
using LlamaCpp.Bindings.Native.SafeHandles;

namespace LlamaCpp.Bindings;

/// <summary>
/// A token-sampling pipeline built from llama.cpp's chain-based sampler API.
/// Construct via <see cref="LlamaSamplerBuilder"/>. Stateful — keeps per-step
/// history for penalties, RNG state for distribution, etc. A single sampler
/// corresponds to one conversation's decode loop; create a new one for a new
/// conversation (or call <see cref="Reset"/> between reuses).
/// </summary>
public sealed class LlamaSampler : IDisposable
{
    private readonly SafeLlamaSamplerHandle _handle;
    private bool _disposed;

    internal LlamaSampler(SafeLlamaSamplerHandle handle)
    {
        _handle = handle;
    }

    internal IntPtr DangerousHandle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle.DangerousHandle;
        }
    }

    /// <summary>
    /// Pick the next token given the logits produced by the most recent
    /// decode. <paramref name="idx"/> is the position within the decoded
    /// batch to sample from; <c>-1</c> (the default) means "the last
    /// position", which is what you want for single-token generation.
    /// </summary>
    public int Sample(LlamaContext context, int idx = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(context);
        return NativeMethods.llama_sampler_sample(
            _handle.DangerousHandle,
            context.Handle.DangerousHandle,
            idx);
    }

    /// <summary>
    /// Notify the sampler that <paramref name="token"/> was accepted. Required
    /// for stateful samplers (penalties, mirostat) to track history — calling
    /// <c>Sample</c> without a following <c>Accept</c> produces correct
    /// output for stateless chains but breaks penalties.
    /// </summary>
    public void Accept(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_sampler_accept(_handle.DangerousHandle, token);
    }

    /// <summary>Reset per-step state (penalty histories, RNG distribution accumulators).</summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_sampler_reset(_handle.DangerousHandle);
    }

    /// <summary>
    /// Short identifier for the sampler's type (e.g., <c>"chain"</c>, <c>"top-k"</c>,
    /// <c>"dist"</c>). For chains built by <see cref="LlamaSamplerBuilder"/> this
    /// is always <c>"chain"</c>; use <see cref="GetChainStageName"/> to inspect
    /// the inner stages.
    /// </summary>
    public string Name
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var ptr = NativeMethods.llama_sampler_name(_handle.DangerousHandle);
            return ptr == IntPtr.Zero ? string.Empty : (Marshal.PtrToStringUTF8(ptr) ?? string.Empty);
        }
    }

    /// <summary>
    /// Number of stages in this sampler chain. Returns 0 for non-chain samplers
    /// (should not happen via <see cref="LlamaSamplerBuilder"/>, which always
    /// produces chains — this exists for defensive inspection).
    /// </summary>
    public int ChainLength
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.llama_sampler_chain_n(_handle.DangerousHandle);
        }
    }

    /// <summary>
    /// Name of the sampler stage at position <paramref name="index"/> within the
    /// chain (e.g., <c>"top-k"</c>, <c>"temp"</c>, <c>"dist"</c>). Returns null
    /// if the index is out of range or this sampler isn't a chain.
    /// </summary>
    /// <remarks>
    /// Only returns the name, not a handle to the stage — sub-sampler pointers
    /// are owned by the chain and must never be freed by the caller, so we
    /// deliberately don't surface them as managed objects.
    /// </remarks>
    public string? GetChainStageName(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (index < 0 || index >= ChainLength) return null;
        var sub = NativeMethods.llama_sampler_chain_get(_handle.DangerousHandle, index);
        if (sub == IntPtr.Zero) return null;
        var ptr = NativeMethods.llama_sampler_name(sub);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    /// <summary>
    /// Ordered list of stage names in this chain. Convenience wrapper over
    /// <see cref="GetChainStageName"/>.
    /// </summary>
    public IReadOnlyList<string> ChainStageNames
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var n = ChainLength;
            if (n <= 0) return Array.Empty<string>();
            var names = new string[n];
            for (int i = 0; i < n; i++)
            {
                names[i] = GetChainStageName(i) ?? "?";
            }
            return names;
        }
    }

    /// <summary>
    /// RNG seed for this sampler's stochastic stages, or <c>null</c> if the
    /// sampler is deterministic (e.g., a chain ending in <c>greedy</c>).
    /// </summary>
    /// <remarks>
    /// llama.cpp returns <c>LLAMA_DEFAULT_SEED</c> (<c>0xFFFFFFFF</c>) for
    /// samplers without a seeded component — we translate that sentinel to
    /// <c>null</c>. Note the edge case: if you deliberately chose
    /// <c>0xFFFFFFFF</c> as your seed, this property reports null. In
    /// practice nobody does.
    /// </remarks>
    public uint? Seed
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var seed = NativeMethods.llama_sampler_get_seed(_handle.DangerousHandle);
            return seed == 0xFFFFFFFFu ? null : seed;
        }
    }

    /// <summary>
    /// Produce a new sampler that is a deep copy of this one, including all
    /// per-step state (penalty histories, RNG position, etc.). Useful for
    /// branching during speculative decoding, or snapshotting before an
    /// experimental sampling attempt. The clone is independent: disposing
    /// this sampler does not affect the clone and vice versa.
    /// </summary>
    public LlamaSampler Clone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var raw = NativeMethods.llama_sampler_clone(_handle.DangerousHandle);
        if (raw == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_sampler_clone),
                "llama_sampler_clone returned NULL.");
        }
        return new LlamaSampler(SafeLlamaSamplerHandle.FromUnsafeHandle(raw));
    }

    // ----- Performance (Tier 1 expansion) -----

    /// <summary>
    /// Snapshot the sampler chain's timing counters. Only meaningful for
    /// chains; non-chain samplers return a zero-valued snapshot.
    /// </summary>
    public LlamaSamplerPerformance GetPerformance()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var d = NativeMethods.llama_perf_sampler(_handle.DangerousHandle);
        return new LlamaSamplerPerformance(
            SampleMilliseconds: d.t_sample_ms,
            SampleCount:        d.n_sample);
    }

    /// <summary>
    /// Reset the sampler's performance counters to zero.
    /// </summary>
    public void ResetPerformance()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_perf_sampler_reset(_handle.DangerousHandle);
    }

    /// <summary>
    /// Log a human-readable sampler performance report via llama.cpp's log sink.
    /// </summary>
    public void LogPerformanceReport()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMethods.llama_perf_sampler_print(_handle.DangerousHandle);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}

/// <summary>
/// Fluent builder for a <see cref="LlamaSampler"/>. The typical chain ends in
/// a terminal sampler (<c>Distribution</c> for stochastic sampling with a
/// seed, or <c>Greedy</c> for argmax). Truncation samplers (Top-K, Top-P,
/// Min-P) filter the candidate set before the terminal selects.
/// </summary>
/// <remarks>
/// Order matters. Idiomatic chain:
/// <code>
/// new LlamaSamplerBuilder()
///     .WithPenalties(lastN: 64, repeat: 1.1f)   // history-based first
///     .WithTopK(40)                              // then truncation
///     .WithTopP(0.9f)
///     .WithMinP(0.05f)
///     .WithTemperature(0.7f)                     // temperature last before terminal
///     .WithDistribution(seed)                    // terminal selector
///     .Build();
/// </code>
/// Each <c>.WithXxx()</c> method allocates a native sub-sampler and hands
/// ownership to the chain — do not worry about freeing them.
/// </remarks>
public sealed class LlamaSamplerBuilder
{
    private readonly List<IntPtr> _pending = new();
    private bool _hasTerminal;
    private bool _measurePerformance = true;

    /// <summary>
    /// Record per-sampler performance counters. Defaults to true (overriding
    /// llama.cpp's OFF default) so <see cref="LlamaSampler.GetPerformance"/>
    /// returns useful numbers. Call with <c>false</c> if you care about the
    /// sub-microsecond per-sample overhead.
    /// </summary>
    public LlamaSamplerBuilder WithPerformanceMeasurement(bool enabled)
    {
        _measurePerformance = enabled;
        return this;
    }

    /// <summary>Top-K truncation. Keeps only the K highest-probability tokens.</summary>
    public LlamaSamplerBuilder WithTopK(int k)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_top_k(k));
        return this;
    }

    /// <summary>Top-P (nucleus) truncation. Keeps the smallest set of tokens whose cumulative probability ≥ p.</summary>
    public LlamaSamplerBuilder WithTopP(float p, int minKeep = 1)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_top_p(p, (nuint)minKeep));
        return this;
    }

    /// <summary>Min-P truncation. Removes tokens whose probability is below p × max_probability.</summary>
    public LlamaSamplerBuilder WithMinP(float p, int minKeep = 1)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_min_p(p, (nuint)minKeep));
        return this;
    }

    /// <summary>Locally typical sampling.</summary>
    public LlamaSamplerBuilder WithTypical(float p, int minKeep = 1)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_typical(p, (nuint)minKeep));
        return this;
    }

    /// <summary>Apply a temperature. Values &lt; 1 sharpen; &gt; 1 flatten; 0 is greedy.</summary>
    public LlamaSamplerBuilder WithTemperature(float temperature)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_temp(temperature));
        return this;
    }

    /// <summary>
    /// Extended temperature with dynamic stretch — llama.cpp's <c>temp_ext</c>.
    /// <paramref name="delta"/> controls how much the temperature adapts to
    /// the entropy of the current distribution; <paramref name="exponent"/>
    /// shapes the curve. When <paramref name="delta"/> is 0, behaves like
    /// plain <see cref="WithTemperature"/>.
    /// </summary>
    public LlamaSamplerBuilder WithExtendedTemperature(float temperature, float delta, float exponent)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_temp_ext(temperature, delta, exponent));
        return this;
    }

    /// <summary>
    /// XTC (Exclude Top Choices) sampler — when the top candidate exceeds
    /// <paramref name="probability"/>, its probability mass above
    /// <paramref name="threshold"/> is redistributed to lower-ranked tokens.
    /// Encourages less-predictable output without overall flattening.
    /// </summary>
    public LlamaSamplerBuilder WithXtc(float probability, float threshold, int minKeep = 1, uint seed = 0)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_xtc(probability, threshold, (nuint)minKeep, seed));
        return this;
    }

    /// <summary>
    /// Top-nσ sampling — keeps only tokens whose logit is within n standard
    /// deviations of the max logit. Described in Top-nσ: Not All Logits
    /// Are You Need (arXiv:2411.07641).
    /// </summary>
    public LlamaSamplerBuilder WithTopNSigma(float n)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_top_n_sigma(n));
        return this;
    }

    /// <summary>
    /// DRY (Don't Repeat Yourself) repetition avoidance. More sophisticated
    /// than plain repetition penalty — targets multi-token repeated patterns.
    /// </summary>
    /// <param name="vocab">Required — DRY tokenises its sequence breakers.</param>
    /// <param name="contextTrainSize">Training context size of the model (from <see cref="LlamaModel.TrainingContextSize"/>).</param>
    /// <param name="multiplier">Penalty strength. <c>0.8</c> is a typical default.</param>
    /// <param name="dryBase">Penalty curve base. <c>1.75</c> is a typical default.</param>
    /// <param name="allowedLength">Pattern length tolerated before penalty kicks in.</param>
    /// <param name="penaltyLastN">Look-back window. <c>-1</c> = full context.</param>
    /// <param name="sequenceBreakers">Strings that reset DRY's internal pattern tracker (e.g., newline, period). Null = defaults.</param>
    public LlamaSamplerBuilder WithDry(
        LlamaVocab vocab,
        int contextTrainSize,
        float multiplier = 0.8f,
        float dryBase = 1.75f,
        int allowedLength = 2,
        int penaltyLastN = -1,
        IReadOnlyList<string>? sequenceBreakers = null)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ThrowIfTerminalAdded();

        var breakers = sequenceBreakers ?? Array.Empty<string>();
        var handles = new IntPtr[breakers.Count];
        try
        {
            for (int i = 0; i < breakers.Count; i++)
            {
                handles[i] = Marshal.StringToCoTaskMemUTF8(breakers[i]);
            }
            unsafe
            {
                fixed (IntPtr* ptrArr = handles)
                {
                    var sub = NativeMethods.llama_sampler_init_dry(
                        vocab.Handle, contextTrainSize,
                        multiplier, dryBase, allowedLength, penaltyLastN,
                        ptrArr, (nuint)breakers.Count);
                    _pending.Add(sub);
                }
            }
        }
        finally
        {
            // llama.cpp copies the breaker strings internally; safe to release now.
            foreach (var h in handles)
            {
                if (h != IntPtr.Zero) Marshal.FreeCoTaskMem(h);
            }
        }
        return this;
    }

    /// <summary>
    /// FIM (fill-in-the-middle) infill sampler — used after top_k+top_p for
    /// code-completion tasks. Does not apply to chat/instruct models.
    /// Requires a model with FIM tokens defined.
    /// </summary>
    public LlamaSamplerBuilder WithInfill(LlamaVocab vocab)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_infill(vocab.Handle));
        return this;
    }

    /// <summary>
    /// Terminal Mirostat v1 sampler — adaptive perplexity control (see
    /// arXiv:2007.14966). Targets a specific "surprise" level across
    /// generation. Prefer <see cref="WithMirostatV2"/> unless you specifically
    /// need v1's <c>m</c> parameter.
    /// </summary>
    public LlamaSamplerBuilder WithMirostat(int vocabSize, uint seed, float tau, float eta, int m = 100)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_mirostat(vocabSize, seed, tau, eta, m));
        _hasTerminal = true;
        return this;
    }

    /// <summary>
    /// Terminal Mirostat v2 sampler — simpler and usually preferred over v1.
    /// Default parameters (τ=5, η=0.1) are a reasonable starting point.
    /// </summary>
    public LlamaSamplerBuilder WithMirostatV2(uint seed, float tau = 5.0f, float eta = 0.1f)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_mirostat_v2(seed, tau, eta));
        _hasTerminal = true;
        return this;
    }

    /// <summary>
    /// Terminal adaptive-p sampler — maintains an EMA of selected-token
    /// probabilities and steers toward a target probability. Use with only
    /// mild prior truncation (e.g., min-p). See upstream PR ggml-org/llama.cpp#17927.
    /// </summary>
    /// <param name="target">Target probability (0..1). Negative = disabled.</param>
    /// <param name="decay">EMA decay (0..0.99). Higher = longer memory.</param>
    public LlamaSamplerBuilder WithAdaptiveP(float target, float decay, uint seed)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_adaptive_p(target, decay, seed));
        _hasTerminal = true;
        return this;
    }

    /// <summary>
    /// Repetition / frequency / presence penalties. <paramref name="lastN"/>
    /// is the look-back window (<c>-1</c> = context size, <c>0</c> = disable).
    /// Values of 1.0 / 0.0 / 0.0 are the "off" defaults.
    /// </summary>
    public LlamaSamplerBuilder WithPenalties(
        int lastN = 64,
        float repeat = 1.0f,
        float frequency = 0.0f,
        float presence = 0.0f)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_penalties(lastN, repeat, frequency, presence));
        return this;
    }

    /// <summary>
    /// Terminal stochastic sampler. Seeds the RNG; identical seeds produce
    /// identical outputs for identical prompt + param combinations.
    /// </summary>
    public LlamaSamplerBuilder WithDistribution(uint seed)
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_dist(seed));
        _hasTerminal = true;
        return this;
    }

    /// <summary>Terminal greedy sampler. Always picks the argmax token — deterministic.</summary>
    public LlamaSamplerBuilder WithGreedy()
    {
        ThrowIfTerminalAdded();
        _pending.Add(NativeMethods.llama_sampler_init_greedy());
        _hasTerminal = true;
        return this;
    }

    /// <summary>
    /// Materialise the sampler. After this, the builder is drained — call
    /// <c>Build</c> again on a fresh builder for a new sampler.
    /// </summary>
    public LlamaSampler Build()
    {
        LlamaBackend.EnsureInitialized();

        if (_pending.Count == 0)
        {
            throw new InvalidOperationException(
                "Sampler has no stages. Add at least a terminal (e.g. .WithDistribution(seed) or .WithGreedy()).");
        }
        if (!_hasTerminal)
        {
            throw new InvalidOperationException(
                "Sampler chain is missing a terminal stage. End with .WithDistribution(seed) or .WithGreedy().");
        }

        var chainParams = NativeMethods.llama_sampler_chain_default_params();
        chainParams.no_perf = !_measurePerformance;
        var chain = NativeMethods.llama_sampler_chain_init(chainParams);
        if (chain == IntPtr.Zero)
        {
            // Failed before ownership transfer — we still own every pending sampler.
            FreePending();
            throw new LlamaException(
                nameof(NativeMethods.llama_sampler_chain_init),
                "llama_sampler_chain_init returned NULL.");
        }

        // Transfer every pending sub-sampler into the chain. After add, the
        // chain owns them — we clear _pending so dispose/dispose-on-throw
        // doesn't double-free.
        foreach (var sub in _pending)
        {
            NativeMethods.llama_sampler_chain_add(chain, sub);
        }
        _pending.Clear();
        _hasTerminal = false;

        return new LlamaSampler(SafeLlamaSamplerHandle.FromUnsafeHandle(chain));
    }

    private void ThrowIfTerminalAdded()
    {
        if (_hasTerminal)
        {
            throw new InvalidOperationException(
                "Terminal stage already added — .WithDistribution/.WithGreedy must be last.");
        }
    }

    private void FreePending()
    {
        foreach (var ptr in _pending)
        {
            if (ptr != IntPtr.Zero)
            {
                NativeMethods.llama_sampler_free(ptr);
            }
        }
        _pending.Clear();
    }
}
