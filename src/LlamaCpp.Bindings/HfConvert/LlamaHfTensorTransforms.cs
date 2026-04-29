using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.HfConvert;

/// <summary>
/// Library of named tensor transforms invoked by the converter engine.
/// V1 ships <c>passthrough</c> — copy bytes through with dtype
/// conversion to the chosen output type. Future architectures that
/// need reshape/permute/pack logic will land their transforms here.
/// </summary>
public static class LlamaHfTensorTransforms
{
    /// <summary>
    /// Identity transform with dtype conversion. Returns the output
    /// bytes plus the GGUF type id (<c>ggml_type</c> integer).
    /// </summary>
    public static (byte[] outBytes, uint outTypeId) Passthrough(
        byte[] sourceBytes,
        LlamaSafetensorsDtype sourceDtype,
        LlamaHfConvertOutputType outputType)
    {
        return (sourceDtype, outputType) switch
        {
            // Same-type passthrough — just hand the bytes back.
            (LlamaSafetensorsDtype.F32,  LlamaHfConvertOutputType.F32)  => (sourceBytes, 0u),
            (LlamaSafetensorsDtype.F16,  LlamaHfConvertOutputType.F16)  => (sourceBytes, 1u),
            (LlamaSafetensorsDtype.BF16, LlamaHfConvertOutputType.BF16) => (sourceBytes, 30u),

            // Up/down convert. We always go through F32 for simplicity —
            // this is one extra allocation but avoids 6 specialized paths.
            // For very large tensors we'd want a streaming variant; V1
            // accepts the peak memory cost.
            _ => ConvertViaF32(sourceBytes, sourceDtype, outputType),
        };
    }

    private static (byte[] outBytes, uint outTypeId) ConvertViaF32(
        byte[] sourceBytes, LlamaSafetensorsDtype sourceDtype, LlamaHfConvertOutputType outputType)
    {
        var f32 = ToFloat32(sourceBytes, sourceDtype);
        return EncodeF32(f32, outputType);
    }

    /// <summary>
    /// Llama-style "undo HF permute" on a Q or K projection weight.
    /// HF stores the rotary halves of each head interleaved row-wise;
    /// llama.cpp's RoPE expects them un-interleaved. The transform
    /// reads the 2-D tensor as <c>[n_groups, 2, half_size, in_features]</c>
    /// (F32 view), swaps the middle two axes to
    /// <c>[n_groups, half_size, 2, in_features]</c>, and re-encodes to
    /// the requested output dtype.
    /// </summary>
    /// <param name="sourceBytes">Source tensor bytes (row-major, HF safetensors layout).</param>
    /// <param name="sourceDtype">Source dtype (F32, F16, or BF16).</param>
    /// <param name="outputType">Output dtype.</param>
    /// <param name="shape">Source tensor shape, row-major (HF order: <c>[out_features, in_features]</c>).</param>
    /// <param name="nHead">Attention head count (for Q) or KV head count (for K). See remarks.</param>
    /// <param name="nHeadKv">KV head count when GQA is in use; <c>null</c> means MHA (n_head_kv == n_head).</param>
    /// <param name="isK">When true, applies the K-projection grouping rule (<c>n_head //= n_head_kv</c> under GQA); when false, treats the tensor as Q.</param>
    /// <remarks>
    /// <para>Mirrors upstream <c>convert_hf_to_gguf.py</c>'s
    /// <c>LlamaModel.permute</c> (the @staticmethod actually called from
    /// <c>LlamaModel.modify_tensors</c>):</para>
    /// <code>
    /// if n_head_kv is not None and n_head != n_head_kv:
    ///     n_head = n_head_kv     # REPLACE (not n_head //= n_kv_head)
    /// return weights.reshape(n_head, 2, weights.shape[0]//n_head//2, *rest).swapaxes(1,2).reshape(weights.shape)
    /// </code>
    /// <para>The call site passes <c>permute(weights, n_head, n_head)</c>
    /// for Q (the conditional is false, n_head unchanged) and
    /// <c>permute(weights, n_head, n_kv_head)</c> for K (under GQA, the
    /// K projection's outer dimension factors as <c>n_kv_head × 2 ×
    /// half_head_dim</c>, not <c>n_head × 2 × half_head_dim</c>). Note
    /// that Llama's <c>permute</c> differs from the older
    /// <c>_reverse_hf_permute</c> used by some other architectures —
    /// that variant divides instead of replacing — and produces a
    /// different K layout. Mixing the two yields tensor data that's
    /// shape-correct but byte-wrong; only the static <c>permute</c>
    /// matches what llama.cpp's RoPE expects for Llama checkpoints.</para>
    /// </remarks>
    public static (byte[] outBytes, uint outTypeId) PermuteQK(
        byte[] sourceBytes,
        LlamaSafetensorsDtype sourceDtype,
        LlamaHfConvertOutputType outputType,
        long[] shape,
        int nHead,
        int? nHeadKv,
        bool isK)
    {
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Length < 2)
            throw new ArgumentException(
                $"PermuteQK expects a 2-D tensor (shape len ≥ 2); got rank {shape.Length}.", nameof(shape));

        // Mirror upstream LlamaModel.permute: under GQA, the K
        // projection's outer dim factors as n_kv_head × 2 × half_head_dim,
        // so n_head is REPLACED by n_kv_head (not divided by it).
        int nGroups = nHead;
        if (isK && nHeadKv is int kv && nHead != kv)
        {
            nGroups = kv;
        }

        long outFeatures = shape[0];
        if (nGroups <= 0 || (outFeatures % (2L * nGroups)) != 0)
            throw new InvalidOperationException(
                $"PermuteQK: shape[0]={outFeatures} not divisible by 2×{nGroups}.");
        long halfSize = outFeatures / nGroups / 2;

        long restElements = 1;
        for (int i = 1; i < shape.Length; i++) restElements *= shape[i];
        if (restElements > int.MaxValue)
            throw new InvalidOperationException(
                $"PermuteQK: per-row element count {restElements} exceeds int.MaxValue.");

        var f32Source = ToFloat32(sourceBytes, sourceDtype);
        if ((long)f32Source.Length != outFeatures * restElements)
            throw new InvalidDataException(
                $"PermuteQK: decoded element count {f32Source.Length} disagrees with shape product {outFeatures * restElements}.");

        var f32Permuted = new float[f32Source.Length];
        int rowFloats = (int)restElements;
        for (long h = 0; h < nGroups; h++)
        {
            for (long half = 0; half < 2; half++)
            {
                for (long i = 0; i < halfSize; i++)
                {
                    // Source layout (HF): [h, half, i, *rest]   — half then i
                    // Output layout (ggml): [h, i, half, *rest] — i then half
                    long srcRow = (h * 2 + half) * halfSize + i;
                    long dstRow = (h * halfSize + i) * 2 + half;
                    Array.Copy(
                        f32Source, srcRow * restElements,
                        f32Permuted, dstRow * restElements,
                        rowFloats);
                }
            }
        }

        return EncodeF32(f32Permuted, outputType);
    }

    private static (byte[] outBytes, uint outTypeId) EncodeF32(float[] f32, LlamaHfConvertOutputType outputType) =>
        outputType switch
        {
            LlamaHfConvertOutputType.F32  => (MemoryMarshal.AsBytes(f32.AsSpan()).ToArray(), 0u),
            LlamaHfConvertOutputType.F16  => (FromFloat32ToF16(f32),  1u),
            LlamaHfConvertOutputType.BF16 => (FromFloat32ToBf16(f32), 30u),
            _ => throw new ArgumentOutOfRangeException(nameof(outputType), outputType, null),
        };

    /// <summary>Decode a tensor's raw bytes to a flat F32 array.</summary>
    private static float[] ToFloat32(byte[] bytes, LlamaSafetensorsDtype dtype)
    {
        switch (dtype)
        {
            case LlamaSafetensorsDtype.F32:
            {
                if ((bytes.Length & 3) != 0)
                    throw new InvalidDataException($"F32 tensor byte length {bytes.Length} not divisible by 4.");
                var arr = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
                return arr;
            }
            case LlamaSafetensorsDtype.F16:
            {
                if ((bytes.Length & 1) != 0)
                    throw new InvalidDataException($"F16 tensor byte length {bytes.Length} not divisible by 2.");
                int n = bytes.Length / 2;
                var arr = new float[n];
                var src = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
                for (int i = 0; i < n; i++)
                {
                    // .NET 7+: Half(ushort bits) via BitConverter.UInt16BitsToHalf.
                    arr[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                }
                return arr;
            }
            case LlamaSafetensorsDtype.BF16:
            {
                if ((bytes.Length & 1) != 0)
                    throw new InvalidDataException($"BF16 tensor byte length {bytes.Length} not divisible by 2.");
                int n = bytes.Length / 2;
                var arr = new float[n];
                var src = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
                for (int i = 0; i < n; i++)
                {
                    // BF16 → F32: shift the 16 bits into the high half
                    // of a 32-bit IEEE float. Identical exponent layout
                    // means no extra translation is needed.
                    uint hi = (uint)src[i] << 16;
                    arr[i] = BitConverter.UInt32BitsToSingle(hi);
                }
                return arr;
            }
            default:
                throw new NotSupportedException(
                    $"Cannot convert dtype {dtype} to F32 — only F32, F16, BF16 supported in V1.");
        }
    }

    /// <summary>Pack an F32 array as F16 bytes.</summary>
    private static byte[] FromFloat32ToF16(float[] f32)
    {
        var bytes = new byte[f32.Length * 2];
        var dst = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
        for (int i = 0; i < f32.Length; i++)
        {
            dst[i] = BitConverter.HalfToUInt16Bits((Half)f32[i]);
        }
        return bytes;
    }

    /// <summary>
    /// Pack an F32 array as BF16 bytes. We use round-to-nearest-even on
    /// the truncated mantissa for fidelity with the upstream Python
    /// converter; a flat truncation is faster but introduces a
    /// systematic downward bias.
    /// </summary>
    private static byte[] FromFloat32ToBf16(float[] f32)
    {
        var bytes = new byte[f32.Length * 2];
        var dst = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
        for (int i = 0; i < f32.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(f32[i]);
            // Round-to-nearest-even on the discarded 16 mantissa bits.
            uint rounding = 0x00007FFFu + ((bits >> 16) & 1u);
            uint rounded  = bits + rounding;
            // NaN should stay NaN. If exponent is all-ones we keep the
            // top mantissa bit nonzero so the result remains a NaN
            // rather than collapsing to ±Inf.
            if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0)
            {
                rounded = bits;
            }
            dst[i] = (ushort)(rounded >> 16);
        }
        return bytes;
    }
}
