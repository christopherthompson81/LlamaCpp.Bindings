using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// A control vector — a per-layer steering direction blended into model
/// activations during inference. Loaded from a GGUF file shaped like
/// llama.cpp's control-vector training output: one F32 1-D tensor per
/// layer named <c>direction.&lt;layer_idx&gt;</c> (1-indexed; layer 0 is
/// reserved and skipped).
/// </summary>
/// <remarks>
/// <para>
/// Lifetime: the vector holds only managed data (a flat <see cref="Data"/>
/// array sized <see cref="NEmbd"/> × <see cref="LayerCount"/>) so it can
/// be freely reused across contexts and disposed at any time.
/// </para>
/// <para>
/// To apply a vector, call <c>LlamaContext.SetControlVector(vec, start, end)</c>.
/// Detach via <c>LlamaContext.ClearControlVector()</c>. Layer ranges are
/// inclusive bounds applied at attach time; the same vector can be
/// reapplied with a different range without reloading the file.
/// </para>
/// </remarks>
public sealed class LlamaControlVector
{
    /// <summary>
    /// Embedding dimension — the length of each per-layer direction.
    /// All directions in a single vector must agree on this value.
    /// </summary>
    public int NEmbd { get; }

    /// <summary>
    /// Number of layers covered (the highest layer index in the source
    /// file). Layer 1 occupies <c>Data[0..NEmbd)</c>, layer 2 occupies
    /// <c>Data[NEmbd..2*NEmbd)</c>, and so on. Layers absent from the
    /// source file are zero-filled.
    /// </summary>
    public int LayerCount => Data.Length / NEmbd;

    /// <summary>
    /// Flat per-layer direction data. Length = <see cref="NEmbd"/> ×
    /// <see cref="LayerCount"/>. Exposed for advanced consumers; most
    /// callers should treat the value as opaque.
    /// </summary>
    public float[] Data { get; }

    internal LlamaControlVector(int nEmbd, float[] data)
    {
        NEmbd = nEmbd;
        Data = data;
    }

    /// <summary>
    /// Load a control vector from a GGUF file. <paramref name="scale"/>
    /// multiplies every direction component during load — pre-scaling the
    /// data is cheaper than rescaling at attach time.
    /// </summary>
    /// <exception cref="FileNotFoundException">File does not exist.</exception>
    /// <exception cref="LlamaException">
    /// GGUF parse error, no <c>direction.*</c> tensors, mixed embedding
    /// dimensions, or non-F32 tensors.
    /// </exception>
    public static LlamaControlVector LoadFromFile(string path, float scale = 1.0f)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Control-vector file not found: {path}", path);
        }

        LlamaBackend.EnsureInitialized();

        // no_alloc=true keeps the tensor data on disk; we read what we
        // need at the offsets ggml reports. Avoids pulling in the
        // ggml_context allocator API.
        var initParams = new gguf_init_params { no_alloc = true, ctx = IntPtr.Zero };
        var gguf = NativeMethods.gguf_init_from_file(path, initParams);
        if (gguf == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.gguf_init_from_file),
                $"Failed to parse GGUF '{path}'. Check the native log for details.");
        }

        try
        {
            long nTensors = NativeMethods.gguf_get_n_tensors(gguf);
            ulong dataOffset = (ulong)NativeMethods.gguf_get_data_offset(gguf);

            var perLayer = new Dictionary<int, float[]>();
            int nEmbd = -1;

            // We open the file once and seek per direction tensor. The
            // FileStream is closed via the using block; gguf_free is
            // called in the outer finally regardless of how this exits.
            using var file = File.OpenRead(path);

            for (long i = 0; i < nTensors; i++)
            {
                var namePtr = NativeMethods.gguf_get_tensor_name(gguf, i);
                var name = Marshal.PtrToStringUTF8(namePtr) ?? string.Empty;

                // Direction tensor naming convention: "direction.<idx>"
                // where idx >= 1. Tensors that don't match the pattern
                // are skipped (the file may contain metadata tensors).
                if (!name.StartsWith("direction.", StringComparison.Ordinal)) continue;
                if (!int.TryParse(
                        name.AsSpan("direction.".Length),
                        System.Globalization.NumberStyles.Integer,
                        System.Globalization.CultureInfo.InvariantCulture,
                        out int layerIdx))
                {
                    continue;
                }
                if (layerIdx < 1)
                {
                    throw new LlamaException(
                        nameof(NativeMethods.gguf_init_from_file),
                        $"Control vector '{path}' has invalid layer index {layerIdx} " +
                        "in tensor name '" + name + "' — layer 0 is reserved.");
                }

                var type = NativeMethods.gguf_get_tensor_type(gguf, i);
                if (type != ggml_type.GGML_TYPE_F32)
                {
                    throw new LlamaException(
                        nameof(NativeMethods.gguf_get_tensor_type),
                        $"Control vector '{path}' tensor '{name}' is type {type}; expected F32.");
                }

                ulong size = (ulong)NativeMethods.gguf_get_tensor_size(gguf, i);
                ulong offset = (ulong)NativeMethods.gguf_get_tensor_offset(gguf, i);
                int elementCount = (int)(size / sizeof(float));

                if (nEmbd == -1)
                {
                    nEmbd = elementCount;
                }
                else if (nEmbd != elementCount)
                {
                    throw new LlamaException(
                        nameof(NativeMethods.gguf_get_tensor_size),
                        $"Control vector '{path}' has inconsistent embedding dimensions: " +
                        $"layer {layerIdx} tensor has {elementCount} elements, expected {nEmbd}.");
                }

                var bytes = new byte[(int)size];
                file.Seek((long)(dataOffset + offset), SeekOrigin.Begin);
                file.ReadExactly(bytes);

                var floats = new float[elementCount];
                Buffer.BlockCopy(bytes, 0, floats, 0, (int)size);

                if (scale != 1.0f)
                {
                    for (int j = 0; j < floats.Length; j++) floats[j] *= scale;
                }

                // Allow a single file to contribute to one layer once.
                // Llama.cpp's loader does sum-merge multiple tensors per
                // layer (e.g. if the file has both "direction.1" and a
                // duplicate); we follow the same accumulation rule.
                if (perLayer.TryGetValue(layerIdx, out var existing))
                {
                    for (int j = 0; j < existing.Length; j++) existing[j] += floats[j];
                }
                else
                {
                    perLayer[layerIdx] = floats;
                }
            }

            if (perLayer.Count == 0 || nEmbd <= 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.gguf_init_from_file),
                    $"Control vector '{path}' has no valid direction.<idx> tensors.");
            }

            int maxLayer = 0;
            foreach (var k in perLayer.Keys)
            {
                if (k > maxLayer) maxLayer = k;
            }
            // Flat layout: layer 1 at index [0, nEmbd), layer 2 at
            // [nEmbd, 2*nEmbd), etc. Layers absent from the file stay
            // zero-filled — that's how llama.cpp's reference loader
            // builds the buffer too.
            var data = new float[maxLayer * nEmbd];
            foreach (var (layer, layerData) in perLayer)
            {
                Array.Copy(layerData, 0, data, (layer - 1) * nEmbd, nEmbd);
            }

            return new LlamaControlVector(nEmbd, data);
        }
        finally
        {
            NativeMethods.gguf_free(gguf);
        }
    }

    /// <summary>
    /// Combine two control vectors by element-wise sum. Both must agree on
    /// <see cref="NEmbd"/>; the result's <see cref="LayerCount"/> is the
    /// max of the two operands (the shorter vector is implicitly zero-
    /// padded). Useful for stacking multiple steering directions on the
    /// same model — applying ten files at once is cheap once their data
    /// is merged.
    /// </summary>
    public LlamaControlVector Combine(LlamaControlVector other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (NEmbd != other.NEmbd)
        {
            throw new ArgumentException(
                $"Cannot combine control vectors with different n_embd " +
                $"({NEmbd} vs {other.NEmbd}). Pair vectors trained against the same model.",
                nameof(other));
        }

        int max = Math.Max(Data.Length, other.Data.Length);
        var merged = new float[max];
        for (int i = 0; i < Data.Length; i++) merged[i] = Data[i];
        for (int i = 0; i < other.Data.Length; i++) merged[i] += other.Data[i];
        return new LlamaControlVector(NEmbd, merged);
    }
}
