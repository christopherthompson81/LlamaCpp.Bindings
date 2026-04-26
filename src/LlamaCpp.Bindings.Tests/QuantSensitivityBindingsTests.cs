using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tests for the new low-level quantization bindings that the
/// Adaptive Quantization sensitivity sweep depends on:
/// <list type="bullet">
///   <item><c>ggml_quantize_chunk</c> — quantize F32 → target type.</item>
///   <item><c>ggml_get_type_traits</c> — fetch the per-type
///         <c>to_float</c> callback for the dequantize half.</item>
///   <item><see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>
///         — pinned per-tensor type overrides flowing through to the
///         native struct's <c>tt_overrides</c> field.</item>
/// </list>
/// </summary>
public class QuantSensitivityBindingsTests
{
    /// <summary>
    /// Quantize a known F32 buffer to Q8_0, dequantize via the type
    /// traits' <c>to_float</c>, and verify the round-trip is within
    /// Q8_0's expected per-element error. If either binding is wrong
    /// the values come back garbled.
    /// </summary>
    [Fact]
    public unsafe void Quantize_Then_Dequantize_Q8_0_Round_Trip_Within_Tolerance()
    {
        LlamaBackend.Initialize();

        // Q8_0 has block size 32 and 34 bytes/block (1 fp16 scale + 32 int8s).
        // Use 256 elements = 8 blocks for a non-trivial sample.
        const int N = 256;
        var src = new float[N];
        for (int i = 0; i < N; i++) src[i] = (float)Math.Sin(i * 0.1) * 0.5f;

        var traitsPtr = NativeMethods.ggml_get_type_traits(ggml_type.GGML_TYPE_Q8_0);
        Assert.NotEqual(IntPtr.Zero, traitsPtr);
        var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);
        Assert.True(traits.is_quantized);
        Assert.Equal(32, traits.blck_size);
        Assert.Equal((nuint)34, traits.type_size);

        long blocks = N / traits.blck_size;
        long byteCount = blocks * (long)traits.type_size;
        var dst = new byte[byteCount];

        long written;
        fixed (float* psrc = src)
        fixed (byte* pdst = dst)
        {
            written = (long)NativeMethods.ggml_quantize_chunk(
                ggml_type.GGML_TYPE_Q8_0, psrc, pdst,
                start: 0, nrows: 1, n_per_row: N, imatrix: null);
        }
        Assert.Equal(byteCount, written);

        // Dequantize via the type-traits function pointer.
        var toFloat = Marshal.GetDelegateForFunctionPointer<GgmlToFloat>(traits.to_float);
        var roundTripped = new float[N];
        fixed (byte* pq = dst)
        fixed (float* pr = roundTripped)
        {
            toFloat((IntPtr)pq, (IntPtr)pr, N);
        }

        // Q8_0 with values in [-0.5, 0.5] has per-element error well
        // below 1/127 of the block scale (~0.5 / 127 ≈ 0.004). Allow
        // 0.02 for safety — far enough below the signal that a wrong
        // binding (garbled bytes, wrong stride) would blow past it.
        double maxAbsErr = 0;
        for (int i = 0; i < N; i++)
        {
            double err = Math.Abs(src[i] - roundTripped[i]);
            if (err > maxAbsErr) maxAbsErr = err;
        }
        Assert.True(maxAbsErr < 0.02,
            $"Q8_0 round-trip max abs error {maxAbsErr:F6} exceeds tolerance — likely binding bug.");
    }

    /// <summary>
    /// Imatrix path: pass a column-importance vector and verify the
    /// call still succeeds. Doesn't assert on improved error vs.
    /// unweighted (the win is statistical, not always reproducible
    /// on a 256-element synthetic), just that the binding accepts the
    /// optional parameter without crashing.
    /// </summary>
    [Fact]
    public unsafe void Quantize_Chunk_Accepts_Imatrix_Pointer()
    {
        LlamaBackend.Initialize();

        const int N = 256;
        var src = new float[N];
        for (int i = 0; i < N; i++) src[i] = (float)Math.Sin(i * 0.1);

        var imatrix = new float[N];
        for (int i = 0; i < N; i++) imatrix[i] = 1.0f;  // unweighted equivalent

        var traits = Marshal.PtrToStructure<ggml_type_traits>(
            NativeMethods.ggml_get_type_traits(ggml_type.GGML_TYPE_Q4_K));
        long blocks = N / traits.blck_size;
        var dst = new byte[blocks * (long)traits.type_size];

        long written;
        fixed (float* psrc = src)
        fixed (byte* pdst = dst)
        fixed (float* pimat = imatrix)
        {
            written = (long)NativeMethods.ggml_quantize_chunk(
                ggml_type.GGML_TYPE_Q4_K, psrc, pdst, 0, 1, N, pimat);
        }
        Assert.Equal((long)dst.Length, written);
    }

    /// <summary>
    /// End-to-end check: quantize the cached test model with
    /// <c>ftype=Q4_K_M</c> and a single override pinning
    /// <c>blk.0.attn_q.weight</c> to <c>Q8_0</c>. Verify that one
    /// tensor lands at Q8_0 (our override) while a sibling layer's
    /// equivalent tensor lands at the family default. Confirms the
    /// <c>tt_overrides</c> array marshalling, the regex pattern, and
    /// the round-trip through <c>llama_model_quantize</c>.
    /// </summary>
    /// <remarks>
    /// Upstream gates <c>tt_overrides</c> behind <c>!pure</c>, so we
    /// leave <see cref="LlamaQuantizationParameters.Pure"/> at its
    /// default <c>false</c>. We also escape literal dots in the
    /// regex so the pattern doesn't accidentally widen — see the doc
    /// comment on <see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>.
    /// </remarks>
    [Fact]
    public async Task TensorTypeOverrides_Pin_One_Tensor_End_To_End()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "with-override.gguf");

            await LlamaQuantizer.QuantizeAsync(modelPath, outPath, new LlamaQuantizationParameters
            {
                FileType = LlamaFileType.Q4_K_M,
                AllowRequantize = true,  // source is Q6_K_XL — without this, quantize refuses
                TensorTypeOverrides = new[]
                {
                    new KeyValuePair<string, LlamaTensorType>(
                        @"^blk\.0\.attn_q\.weight$",
                        LlamaTensorType.Q8_0),
                },
            });

            var f = LlamaGgufFile.Open(outPath);
            var pinned = f.Tensors.First(t => t.Name == "blk.0.attn_q.weight");
            Assert.Equal(LlamaTensorType.Q8_0, pinned.Type);

            // Sibling layer's equivalent tensor wasn't matched by our
            // anchored "blk.0" pattern, so it lands at the per-ftype
            // default (a Q4_K-family type for Q4_K_M). Assert only
            // that it ISN'T Q8_0 — exact body type can drift across
            // ggml versions and isn't load-bearing for this test.
            var sibling = f.Tensors.First(t => t.Name == "blk.1.attn_q.weight");
            Assert.NotEqual(LlamaTensorType.Q8_0, sibling.Type);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    /// <summary>Disallow empty pattern keys — they'd terminate the native scanner early and silently drop later overrides.</summary>
    [Fact]
    public async Task TensorTypeOverrides_Empty_Pattern_Key_Throws()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "wont-exist.gguf");
            await Assert.ThrowsAsync<ArgumentException>(() =>
                LlamaQuantizer.QuantizeAsync(modelPath, outPath, new LlamaQuantizationParameters
                {
                    FileType = LlamaFileType.Q4_K_M,
                    Pure = true,
                    AllowRequantize = true,
                    TensorTypeOverrides = new[]
                    {
                        new KeyValuePair<string, LlamaTensorType>("", LlamaTensorType.Q8_0),
                    },
                }));
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-quant-overrides-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void GgmlToFloat(IntPtr src, IntPtr dst, long k);
}
