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
