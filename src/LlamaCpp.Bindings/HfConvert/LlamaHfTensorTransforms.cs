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
        return outputType switch
        {
            LlamaHfConvertOutputType.F32  => (MemoryMarshal.AsBytes(f32.AsSpan()).ToArray(), 0u),
            LlamaHfConvertOutputType.F16  => (FromFloat32ToF16(f32),  1u),
            LlamaHfConvertOutputType.BF16 => (FromFloat32ToBf16(f32), 30u),
            _ => throw new ArgumentOutOfRangeException(nameof(outputType), outputType, null),
        };
    }

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
