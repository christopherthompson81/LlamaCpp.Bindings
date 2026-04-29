using System.Buffers;
using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Simulate quantization of a tensor by round-tripping its data
/// through <c>ggml_quantize_chunk</c> + the type's dequantize step,
/// then re-encoding the dequantized F32 values as F16. The output
/// is the same byte layout and size as the input — what changes is
/// the numerical content, which now matches what real quantization
/// would produce after dequantize-on-the-fly inside the matmul kernel.
/// </summary>
/// <remarks>
/// <para>
/// Designed for the in-place ablator path
/// (<see cref="LlamaInPlaceAblator"/>). It lets us simulate "tensor X
/// at type Q4_K" without writing a new GGUF file or re-loading the
/// model — we keep the F16-allocated tensor in VRAM and just rewrite
/// its bytes with round-tripped values.
/// </para>
/// <para>
/// <strong>Numerical fidelity vs. real quantization:</strong> the
/// dequantized values are bit-identical to what a real Q-quant tensor
/// would produce when its kernel dequantizes it on the fly. The matmul
/// kernel itself differs (F16×input vs Q-kernel×input) and may produce
/// slightly different accumulator orderings, but the difference is
/// within float32 precision noise on the workloads we care about.
/// </para>
/// </remarks>
public static class LlamaTensorRoundTrip
{
    /// <summary>
    /// Round-trip <paramref name="sourceFp16Bytes"/> through quantization
    /// to <paramref name="targetType"/> and back, returning the same byte
    /// length of F16 output suitable for <see cref="LlamaModel.SetTensorData"/>.
    /// </summary>
    /// <param name="sourceFp16Bytes">
    /// Raw F16 tensor data (2 bytes per element). Length must equal
    /// <c>elementCount × 2</c>.
    /// </param>
    /// <param name="targetType">
    /// Quantization type to simulate (e.g. <see cref="LlamaTensorType.Q4_K"/>,
    /// <see cref="LlamaTensorType.IQ4_XS"/>, <see cref="LlamaTensorType.Q6_K"/>).
    /// Must be a quantizable type — F16/F32 returns the input unchanged
    /// after the F16→F32→F16 round trip (sub-ULP noise only).
    /// </param>
    /// <param name="rowCount">Number of rows in the tensor (ggml convention: tensor is shape [colCount, rowCount]).</param>
    /// <param name="colCount">Number of elements per row.</param>
    /// <param name="imatrix">
    /// Optional column-importance matrix of length <paramref name="colCount"/>.
    /// When provided, used by <c>ggml_quantize_chunk</c> for imatrix-aware
    /// quant scaling. Pass null for unweighted.
    /// </param>
    /// <summary>
    /// Allocating overload — returns a fresh <c>byte[]</c> sized to
    /// match the F16 input. Convenient for ad-hoc tests; the
    /// allocation-free <see cref="EncodeInto"/> is the right call for
    /// tight loops (e.g. the in-place ablator's per-cell apply phase).
    /// </summary>
    public static byte[] Encode(
        ReadOnlySpan<byte> sourceFp16Bytes,
        LlamaTensorType targetType,
        long rowCount,
        int colCount,
        ReadOnlySpan<float> imatrix = default)
    {
        var output = new byte[sourceFp16Bytes.Length];
        EncodeInto(sourceFp16Bytes, targetType, rowCount, colCount, imatrix, output);
        return output;
    }

    /// <summary>
    /// Allocation-free variant: writes the round-trip-encoded F16 bytes
    /// into the caller-provided <paramref name="output"/> span. Lets the
    /// ablator (and any other tight-loop caller) keep a per-instance
    /// reusable buffer instead of allocating <c>byte[]</c> per cell.
    /// Internal F32 working buffers are pooled via
    /// <see cref="ArrayPool{T}.Shared"/>.
    /// </summary>
    public static void EncodeInto(
        ReadOnlySpan<byte> sourceFp16Bytes,
        LlamaTensorType targetType,
        long rowCount,
        int colCount,
        ReadOnlySpan<float> imatrix,
        Span<byte> output)
    {
        if (sourceFp16Bytes.Length % 2 != 0)
        {
            throw new ArgumentException(
                "F16 source data length must be even (2 bytes per element).",
                nameof(sourceFp16Bytes));
        }

        long elementCount = (long)rowCount * colCount;
        if (sourceFp16Bytes.Length != elementCount * 2)
        {
            throw new ArgumentException(
                $"sourceFp16Bytes length ({sourceFp16Bytes.Length}) does not match " +
                $"rowCount*colCount*2 ({elementCount * 2}).",
                nameof(sourceFp16Bytes));
        }
        if (!imatrix.IsEmpty && imatrix.Length != colCount)
        {
            throw new ArgumentException(
                $"imatrix length ({imatrix.Length}) must match colCount ({colCount}).",
                nameof(imatrix));
        }
        if (output.Length != sourceFp16Bytes.Length)
        {
            throw new ArgumentException(
                $"output length ({output.Length}) must match sourceFp16Bytes length ({sourceFp16Bytes.Length}).",
                nameof(output));
        }

        // Pool the three large working buffers (F32 source, quant
        // buffer, F32 round-tripped) via ArrayPool. Encoding a 24 MB
        // F16 ffn_down tensor allocates ~140 MB of working memory; with
        // 4 concurrent workers × 28 tensors per cell, fresh allocations
        // build up GC pressure that grows the encode time 4–5× over
        // the course of a campaign. Pooled buffers keep encode time
        // flat at the early ~5–17 s range.
        if (elementCount > int.MaxValue)
        {
            throw new ArgumentException(
                $"Tensor element count ({elementCount}) exceeds Array.MaxLength.",
                nameof(rowCount));
        }
        int n = (int)elementCount;
        var traits = LoadTraits(targetType);
        long quantBytes = ComputeQuantBytes(traits, rowCount, colCount);
        if (quantBytes > int.MaxValue)
        {
            throw new ArgumentException(
                $"Quant buffer size ({quantBytes}) exceeds Array.MaxLength.");
        }

        var fp32SourceArr = ArrayPool<float>.Shared.Rent(n);
        var quantBufArr = ArrayPool<byte>.Shared.Rent((int)quantBytes);
        var fp32RoundTrippedArr = ArrayPool<float>.Shared.Rent(n);
        try
        {
            // Step 1: F16 → F32.
            unsafe
            {
                fixed (byte* srcPtr = sourceFp16Bytes)
                fixed (float* dstPtr = fp32SourceArr)
                {
                    NativeMethods.ggml_fp16_to_fp32_row((ushort*)srcPtr, dstPtr, elementCount);
                }
            }

            // Step 2: quantize F32 → target type.
            unsafe
            {
                fixed (float* srcPtr = fp32SourceArr)
                fixed (byte* dstPtr = quantBufArr)
                fixed (float* imatPtr = imatrix)
                {
                    NativeMethods.ggml_quantize_chunk(
                        (ggml_type)targetType,
                        srcPtr,
                        dstPtr,
                        start: 0,
                        nrows: rowCount,
                        n_per_row: colCount,
                        imatrix: imatrix.IsEmpty ? (float*)null : imatPtr);
                }
            }

            // Step 3: dequantize target type → F32.
            unsafe
            {
                var toFloat = Marshal.GetDelegateForFunctionPointer<NativeMethods.GgmlToFloatDelegate>(traits.to_float);
                fixed (byte* srcPtr = quantBufArr)
                fixed (float* dstPtr = fp32RoundTrippedArr)
                {
                    toFloat((IntPtr)srcPtr, (IntPtr)dstPtr, elementCount);
                }
            }

            // Step 4: F32 → F16 directly into the caller-provided output span.
            unsafe
            {
                fixed (float* srcPtr = fp32RoundTrippedArr)
                fixed (byte* dstPtr = output)
                {
                    NativeMethods.ggml_fp32_to_fp16_row(srcPtr, (ushort*)dstPtr, elementCount);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(fp32SourceArr);
            ArrayPool<byte>.Shared.Return(quantBufArr);
            ArrayPool<float>.Shared.Return(fp32RoundTrippedArr);
        }
    }

    private static ggml_type_traits LoadTraits(LlamaTensorType targetType)
    {
        var traitsPtr = NativeMethods.ggml_get_type_traits((ggml_type)targetType);
        if (traitsPtr == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                $"ggml_get_type_traits returned NULL for type {targetType}.");
        }
        return Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);
    }

    private static long ComputeQuantBytes(ggml_type_traits traits, long rowCount, int colCount)
    {
        if (traits.blck_size <= 1)
        {
            // Non-blocked types (F16/BF16) — type_size bytes per element.
            return rowCount * colCount * (long)traits.type_size;
        }
        if (colCount % traits.blck_size != 0)
        {
            throw new ArgumentException(
                $"colCount ({colCount}) must be a multiple of the type's block size ({traits.blck_size}).",
                nameof(colCount));
        }
        return rowCount * (long)traits.type_size * (colCount / traits.blck_size);
    }
}

