using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.HfConvert;

/// <summary>
/// Generators for tensors that aren't in HF safetensors but must be
/// emitted into the output GGUF. Each generator inspects
/// <see cref="LlamaHfConfig"/> and returns the tensor's bytes + GGUF
/// type id + shape, or <c>null</c> when the config doesn't call for the
/// tensor (e.g., the llama3 RoPE freqs only matter on Llama-3+ with
/// <c>rope_scaling.rope_type == "llama3"</c>; older Llama-2 checkpoints
/// have plain RoPE and don't need the table).
/// </summary>
public static class LlamaHfExtraTensors
{
    /// <summary>Output payload from a generator.</summary>
    public readonly record struct Generated(byte[] Data, uint TypeId, long[] Shape);

    /// <summary>
    /// Compute Llama-3 RoPE frequency adjustment factors. Mirrors
    /// upstream <c>convert_hf_to_gguf.py</c>'s
    /// <c>LlamaModel.generate_extra_tensors</c>:
    /// <code>
    /// freqs = 1 / (base ** (arange(0, dim, 2) / dim))
    /// for each freq:
    ///   wavelen = 2π / freq
    ///   if wavelen &lt; old_ctx / high_freq_factor: factor = 1
    ///   elif wavelen &gt; old_ctx / low_freq_factor: factor = scale_factor
    ///   else: smooth interpolation between 1 and scale_factor
    /// </code>
    /// </summary>
    /// <remarks>
    /// <para>llama.cpp detects llama3-style RoPE by the *presence* of
    /// <c>rope_freqs.weight</c> alone — there's no <c>rope_scaling.type</c>
    /// metadata flag for it (unlike LINEAR / YARN / LONGROPE which all
    /// set explicit type keys). Returning a tensor here is the entire
    /// signal.</para>
    /// <para>Returns null when <c>rope_scaling.rope_type</c> is missing
    /// or isn't <c>"llama3"</c>. Llama-2 checkpoints fall through this
    /// branch; older Llama-3 checkpoints with
    /// <c>rope_scaling.type == "linear"</c> would too (those land on
    /// the LINEAR scaling path which is metadata-only).</para>
    /// </remarks>
    public static Generated? Llama3RopeFreqs(LlamaHfConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        var ropeType = config.GetString("rope_scaling.rope_type")
                    ?? config.GetString("rope_scaling.type");
        if (ropeType is null || !string.Equals(ropeType, "llama3", StringComparison.OrdinalIgnoreCase))
            return null;

        // RoPE base frequency. Note: rope_theta lives at config root,
        // not under rope_scaling. Default 10000.0 is the original RoPE
        // value; Llama-3 uses 500000.0.
        double baseFreq = config.GetFloat64("rope_theta")
                       ?? config.GetFloat64("rope_scaling.rope_theta")
                       ?? 10000.0;

        // head_dim defaults to hidden_size / num_attention_heads.
        int? dim = (int?)config.GetUInt32("head_dim");
        if (dim is null)
        {
            var hidden = config.GetUInt32("hidden_size");
            var nHeads = config.GetUInt32("num_attention_heads");
            if (hidden is null || nHeads is null || nHeads == 0)
                throw new InvalidDataException(
                    "Llama3 RoPE freqs need head_dim, or hidden_size+num_attention_heads to derive it.");
            dim = (int)(hidden.Value / nHeads.Value);
        }
        if (dim <= 0 || (dim & 1) != 0)
            throw new InvalidDataException(
                $"Llama3 RoPE freqs need an even positive head_dim; got {dim}.");

        double scaleFactor    = config.GetFloat64("rope_scaling.factor")             ?? 8.0;
        double lowFreqFactor  = config.GetFloat64("rope_scaling.low_freq_factor")    ?? 1.0;
        double highFreqFactor = config.GetFloat64("rope_scaling.high_freq_factor")   ?? 4.0;
        int    oldCtxLen      = (int?)config.GetUInt32("rope_scaling.original_max_position_embeddings")
                              ?? (int?)config.GetUInt32("original_max_position_embeddings")
                              ?? 8192;

        double lowFreqWavelen  = oldCtxLen / lowFreqFactor;
        double highFreqWavelen = oldCtxLen / highFreqFactor;

        int n = dim.Value / 2;
        var factors = new float[n];
        for (int i = 0; i < n; i++)
        {
            // freqs[i] = 1 / base^(2i / dim). Upstream computes this in
            // F32 (torch.arange dtype=float32, base**... yields float32),
            // then the 2π/freq arithmetic promotes back to double via
            // Python float coercion. Match that evaluation order so the
            // last-bit F32 rounding matches the upstream-converted file
            // byte-for-byte.
            float freqF32 = 1.0f / MathF.Pow((float)baseFreq, (float)((2.0 * i) / dim.Value));
            double freq = freqF32;
            double wavelen = (2.0 * Math.PI) / freq;
            double f;
            if (wavelen < highFreqWavelen)
            {
                f = 1.0;
            }
            else if (wavelen > lowFreqWavelen)
            {
                f = scaleFactor;
            }
            else
            {
                double smooth = ((double)oldCtxLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
                f = 1.0 / ((1.0 - smooth) / scaleFactor + smooth);
            }
            factors[i] = (float)f;
        }

        var bytes = MemoryMarshal.AsBytes(factors.AsSpan()).ToArray();
        return new Generated(bytes, TypeId: 0u /* F32 */, Shape: new long[] { n });
    }
}
