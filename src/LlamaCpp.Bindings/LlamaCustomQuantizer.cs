using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>Tunable knobs for <see cref="LlamaCustomQuantizer.QuantizeWithRecipeAsync"/>.</summary>
public sealed class LlamaCustomQuantizerOptions
{
    /// <summary>Optional imatrix GGUF; per-column importance is forwarded to <c>ggml_quantize_chunk</c>.</summary>
    public string? ImatrixPath { get; set; }

    /// <summary>
    /// When true (default), tensors whose row size is incompatible with
    /// the recipe's chosen K-quant fall back to a compatible legacy
    /// type (matching llama-quant's <c>tensor_type_fallback</c> table).
    /// When false, an incompatible recipe is fatal — useful for tests
    /// that want to assert the recipe is shape-correct.
    /// </summary>
    public bool AllowShapeFallback { get; set; } = true;

    /// <summary>
    /// Skip the well-known "do not quantize" tensors that
    /// <c>tensor_allows_quantization</c> in llama-quant.cpp filters
    /// out: <c>*_norm.weight</c>, expert-gate inputs, positional
    /// embeddings, etc. These pass through at their source type
    /// regardless of any recipe entry. Default true.
    /// </summary>
    public bool ApplyHeuristicSkipList { get; set; } = true;
}

/// <summary>Per-tensor progress for <see cref="LlamaCustomQuantizer"/>.</summary>
public readonly record struct LlamaCustomQuantizerProgress(
    int CompletedTensors,
    int TotalTensors,
    string? CurrentTensor,
    LlamaTensorType? AppliedType);

/// <summary>
/// Custom GGUF quantizer that applies a <see cref="LlamaQuantRecipe"/>
/// exactly — bypassing <c>llama_model_quantize</c>'s built-in heuristic
/// and its "override-only-elevates" rule. The recipe specifies a target
/// type per tensor; this driver realizes those choices verbatim.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists: the stock path runs llama.cpp's <c>use_more_bits</c>
/// heuristic *underneath* the user's <c>tt_overrides</c>, and only fires
/// the override when it differs from the heuristic's pick. Run 14
/// confirmed this means recipe-built recipes silently end up larger
/// than predicted because the heuristic re-promotes ffn_down /
/// output.weight to Q6_K. The custom path produces recipes that match
/// their predicted bpw because every tensor's type is decided by the
/// recipe, full stop.
/// </para>
/// <para>
/// Guardrails replicated from llama-quant:
/// </para>
/// <list type="bullet">
///   <item>1D tensors and norm tensors stay at their source type.</item>
///   <item>Tensors not ending in <c>.weight</c> stay at their source type.</item>
///   <item>Specific known-skip tensors (expert gates, pos embeddings,
///     altup/laurel/per_layer_model_proj small tensors) stay verbatim.</item>
///   <item>K-quants applied to tensors whose row size isn't divisible
///     by the K-quant block size fall back to a compatible legacy type
///     (matching <c>tensor_type_fallback</c>) when
///     <see cref="LlamaCustomQuantizerOptions.AllowShapeFallback"/> is on.</item>
/// </list>
/// </remarks>
public static class LlamaCustomQuantizer
{
    /// <summary>Quantize <paramref name="sourcePath"/> to <paramref name="outputPath"/> per <paramref name="recipe"/>.</summary>
    public static async Task QuantizeWithRecipeAsync(
        string sourcePath,
        string outputPath,
        LlamaQuantRecipe recipe,
        LlamaCustomQuantizerOptions? options = null,
        IProgress<LlamaCustomQuantizerProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        ArgumentNullException.ThrowIfNull(recipe);
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException($"Source not found: {sourcePath}", sourcePath);
        var opts = options ?? new LlamaCustomQuantizerOptions();
        LlamaBackend.EnsureInitialized();

        var src = LlamaGgufFile.Open(sourcePath);
        var writer = new LlamaGgufWriter(alignment: src.Alignment > 0 ? src.Alignment : LlamaGgufWriter.DefaultAlignment);

        // Carry every metadata key over. We never edit the architecture
        // metadata (the recipe doesn't change shapes); the only special
        // case is general.file_type, which downstream tools use to read
        // a coarse "what ftype is this" hint. Mark it MOSTLY_F16 — the
        // closest "bag of mixed types" enum value — and let consumers
        // discover the real per-tensor types from the GGUF tensor table.
        foreach (var m in src.Metadata)
        {
            if (m.Key == "general.file_type")
                writer.SetMetadata(m.Key, (uint)LlamaFileType.MostlyF16);
            else
                writer.SetMetadata(m.Key, m.Value);
        }

        var recipeByName = recipe.Entries.ToDictionary(e => e.TensorName, e => e.ChosenType, StringComparer.Ordinal);
        var imatrix = opts.ImatrixPath is { } imatPath ? LoadImatrix(imatPath) : null;

        int completed = 0;
        int total = src.Tensors.Count;

        for (int i = 0; i < src.Tensors.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var t = src.Tensors[i];
            var srcTypeId = (LlamaTensorType)(int)t.TypeId;

            LlamaTensorType targetType;
            if (!ShouldQuantize(t, opts, recipeByName, out var skipReason))
            {
                targetType = srcTypeId;
            }
            else if (recipeByName.TryGetValue(t.Name, out var recipeType))
            {
                targetType = recipeType;
                if (opts.AllowShapeFallback)
                    targetType = ApplyShapeFallback(t, targetType);
                else
                    EnsureShapeCompatible(t, targetType);
            }
            else
            {
                // Tensor allowed for quantization but recipe doesn't
                // mention it. Conservatively keep its source type rather
                // than guessing. The recipe builder is responsible for
                // emitting an entry for every weight tensor it cares
                // about; uncategorized weights ride at source type.
                targetType = srcTypeId;
            }

            progress?.Report(new LlamaCustomQuantizerProgress(
                CompletedTensors: completed,
                TotalTensors:     total,
                CurrentTensor:    t.Name,
                AppliedType:      targetType));

            if (targetType == srcTypeId)
            {
                // No conversion — copy bytes through. Zero-copy via the
                // writer's file-backed tensor source.
                writer.AddTensorFromFile(
                    name:               t.Name,
                    typeId:             t.TypeId,
                    shape:              t.Dimensions,
                    sourcePath:         sourcePath,
                    sourceOffsetInFile: src.DataSectionFileOffset + t.ByteOffsetInDataSection,
                    byteSize:           t.ByteSize);
            }
            else
            {
                // Dequant source → F32 → requant to targetType.
                var srcF32 = ReadTensorAsFloat32(src, t);
                float[]? imatRow = null;
                if (imatrix is not null && imatrix.TryGetValue(t.Name, out var w))
                    imatRow = w;
                var dstBytes = QuantizeF32(srcF32, targetType, t.Dimensions, imatRow);
                writer.AddTensor(t.Name, (uint)(int)targetType, t.Dimensions, dstBytes);
            }

            completed++;
        }

        progress?.Report(new LlamaCustomQuantizerProgress(
            CompletedTensors: completed,
            TotalTensors:     total,
            CurrentTensor:    null,
            AppliedType:      null));

        await writer.WriteAsync(outputPath, cancellationToken).ConfigureAwait(false);
    }

    // ---- guardrails: which tensors get quantized, and to what -------------

    private static bool ShouldQuantize(
        LlamaGgufTensorInfo t,
        LlamaCustomQuantizerOptions opts,
        IReadOnlyDictionary<string, LlamaTensorType> recipe,
        out string? skipReason)
    {
        skipReason = null;

        // Never quantize 1D tensors. K-quants and most quantized formats
        // require row size divisible by their block size; 1D tensors
        // don't have a row dimension that makes sense.
        if (t.Dimensions.Length < 2) { skipReason = "1D tensor"; return false; }

        if (!opts.ApplyHeuristicSkipList) return true;

        // Mirrors tensor_allows_quantization in llama-quant.cpp:288.
        var name = t.Name;
        if (!name.EndsWith(".weight", StringComparison.Ordinal))
        {
            skipReason = "name does not end with .weight"; return false;
        }
        if (name.Contains("_norm.weight", StringComparison.Ordinal))
        {
            skipReason = "norm tensor"; return false;
        }
        if (name.Contains("ffn_gate_inp.weight", StringComparison.Ordinal))
        {
            skipReason = "expert gating tensor"; return false;
        }
        if (name.Contains("altup", StringComparison.Ordinal) ||
            name.Contains("laurel", StringComparison.Ordinal) ||
            name.Contains("per_layer_model_proj", StringComparison.Ordinal))
        {
            skipReason = "very small / kept-verbatim tensor"; return false;
        }
        if (name.Contains("position_embd", StringComparison.Ordinal) ||
            name.Contains("pos_embd", StringComparison.Ordinal) ||
            name.Contains("token_types", StringComparison.Ordinal))
        {
            skipReason = "positional / type embedding"; return false;
        }
        return true;
    }

    /// <summary>
    /// Mirrors <c>tensor_type_fallback</c> in <c>llama-quant.cpp:362</c>:
    /// when <paramref name="t"/>'s row size isn't divisible by the
    /// chosen type's block size, fall back to a legacy type that the
    /// shape *is* divisible by.
    /// </summary>
    private static LlamaTensorType ApplyShapeFallback(LlamaGgufTensorInfo t, LlamaTensorType target)
    {
        var traitsPtr = NativeMethods.ggml_get_type_traits((ggml_type)(int)target);
        if (traitsPtr == IntPtr.Zero) return target;
        var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);
        long blck = traits.blck_size <= 0 ? 1 : traits.blck_size;
        long ncols = t.Dimensions[0];
        if (ncols % blck == 0) return target;
        return target switch
        {
            LlamaTensorType.IQ4_XS => LlamaTensorType.IQ4_NL,
            LlamaTensorType.Q2_K   => LlamaTensorType.Q4_0,
            LlamaTensorType.Q3_K   => LlamaTensorType.Q4_0,
            LlamaTensorType.Q4_K   => LlamaTensorType.Q5_0,
            LlamaTensorType.Q5_K   => LlamaTensorType.Q5_1,
            LlamaTensorType.Q6_K   => LlamaTensorType.Q8_0,
            _                      => target,
        };
    }

    private static void EnsureShapeCompatible(LlamaGgufTensorInfo t, LlamaTensorType target)
    {
        var traitsPtr = NativeMethods.ggml_get_type_traits((ggml_type)(int)target);
        if (traitsPtr == IntPtr.Zero)
            throw new InvalidOperationException($"No type-traits for {target}.");
        var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);
        long blck = traits.blck_size <= 0 ? 1 : traits.blck_size;
        long ncols = t.Dimensions[0];
        if (ncols % blck != 0)
        {
            throw new InvalidOperationException(
                $"Tensor '{t.Name}' has ncols={ncols}, not divisible by {target}'s block size {blck}. " +
                $"Set Options.AllowShapeFallback=true to fall back to a legacy type, or fix the recipe.");
        }
    }

    // ---- read / write helpers ---------------------------------------------

    private static unsafe float[] ReadTensorAsFloat32(LlamaGgufFile file, LlamaGgufTensorInfo t)
    {
        long elements = 1;
        foreach (var d in t.Dimensions) elements *= d;
        if (elements > int.MaxValue)
            throw new InvalidDataException($"Tensor '{t.Name}' has {elements} elements, exceeding .NET array limit.");
        var arr = new float[(int)elements];

        using var fs = File.OpenRead(file.SourcePath);
        fs.Seek(file.DataSectionFileOffset + t.ByteOffsetInDataSection, SeekOrigin.Begin);
        var raw = new byte[t.ByteSize];
        ReadExactly(fs, raw);

        switch (t.TypeId)
        {
            case 0: // F32
                Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
                break;
            case 1: // F16
            {
                var span = MemoryMarshal.Cast<byte, ushort>(raw.AsSpan());
                for (int i = 0; i < arr.Length; i++)
                    arr[i] = (float)BitConverter.UInt16BitsToHalf(span[i]);
                break;
            }
            case 30: // BF16
            {
                var span = MemoryMarshal.Cast<byte, ushort>(raw.AsSpan());
                for (int i = 0; i < arr.Length; i++)
                {
                    uint hi = (uint)span[i] << 16;
                    arr[i] = BitConverter.UInt32BitsToSingle(hi);
                }
                break;
            }
            default:
            {
                var traitsPtr = NativeMethods.ggml_get_type_traits((ggml_type)t.TypeId);
                if (traitsPtr == IntPtr.Zero)
                    throw new NotSupportedException(
                        $"Tensor '{t.Name}' has type {t.TypeId} with no type-traits — can't dequantize.");
                var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);
                if (traits.to_float == IntPtr.Zero)
                    throw new NotSupportedException(
                        $"Tensor '{t.Name}' (type {t.TypeId}) has no to_float — can't dequantize.");
                var toFloat = Marshal.GetDelegateForFunctionPointer<GgmlToFloatDelegate>(traits.to_float);
                fixed (byte* praw = raw)
                fixed (float* parr = arr)
                {
                    toFloat((IntPtr)praw, (IntPtr)parr, arr.Length);
                }
                break;
            }
        }
        return arr;
    }

    private static unsafe byte[] QuantizeF32(
        float[] src, LlamaTensorType target, long[] dimensions, float[]? imatrix)
    {
        long ncols = dimensions[0];
        long elements = 1;
        foreach (var d in dimensions) elements *= d;
        long nrows = elements / ncols;

        var traitsPtr = NativeMethods.ggml_get_type_traits((ggml_type)(int)target);
        if (traitsPtr == IntPtr.Zero)
            throw new NotSupportedException($"No type-traits for {target}.");
        var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);

        long blck = traits.blck_size <= 0 ? 1 : traits.blck_size;
        if (ncols % blck != 0)
            throw new InvalidOperationException(
                $"ncols {ncols} not divisible by block size {blck} for {target} — apply shape fallback first.");

        // Output buffer: rows × type_size bytes per (block_size elements).
        long bytesPerRow = (ncols / blck) * (long)traits.type_size;
        long totalBytes = nrows * bytesPerRow;
        if (totalBytes > int.MaxValue)
            throw new InvalidDataException(
                $"Quantized tensor would be {totalBytes} bytes, exceeding .NET array limit.");
        var dst = new byte[totalBytes];

        if (imatrix is not null && imatrix.Length != ncols)
        {
            // Imatrix vector dimension mismatch — drop it rather than apply incorrectly.
            imatrix = null;
        }

        fixed (float* psrc = src)
        fixed (byte*  pdst = dst)
        {
            if (imatrix is not null)
            {
                fixed (float* pimat = imatrix)
                {
                    NativeMethods.ggml_quantize_chunk(
                        (ggml_type)(int)target, psrc, pdst,
                        start: 0, nrows: nrows, n_per_row: ncols,
                        imatrix: pimat);
                }
            }
            else
            {
                NativeMethods.ggml_quantize_chunk(
                    (ggml_type)(int)target, psrc, pdst,
                    start: 0, nrows: nrows, n_per_row: ncols,
                    imatrix: null);
            }
        }
        return dst;
    }

    /// <summary>
    /// Load an imatrix GGUF and return per-tensor column-importance
    /// vectors. Same format as <see cref="LlamaQuantSensitivity"/>:
    /// each tracked weight has <c>&lt;name&gt;.in_sum2</c> (squared-sum
    /// F32) and <c>&lt;name&gt;.counts</c>; importance =
    /// <c>in_sum2 / counts</c>, averaged across the nmat axis for
    /// dense tensors.
    /// </summary>
    private static Dictionary<string, float[]> LoadImatrix(string path)
    {
        var f = LlamaGgufFile.Open(path);
        var byBaseName = new Dictionary<string, float[]>(StringComparer.Ordinal);
        const string SumSuffix = ".in_sum2";

        foreach (var t in f.Tensors)
        {
            if (!t.Name.EndsWith(SumSuffix, StringComparison.Ordinal)) continue;
            string baseName = t.Name[..^SumSuffix.Length];
            string countsName = baseName + ".counts";
            var counts = f.Tensors.FirstOrDefault(x => x.Name == countsName);
            if (counts is null) continue;

            var sumsBytes = ReadRawBytes(f, t);
            var countsBytes = ReadRawBytes(f, counts);
            int sumCount = sumsBytes.Length / sizeof(float);
            int countsCount = countsBytes.Length / sizeof(float);
            if (countsCount == 0) continue;

            int nEmbd = sumCount / countsCount;
            if (nEmbd * countsCount != sumCount) continue;

            var sumsArr = new float[sumCount];
            Buffer.BlockCopy(sumsBytes, 0, sumsArr, 0, sumsBytes.Length);
            var countsArr = new float[countsCount];
            Buffer.BlockCopy(countsBytes, 0, countsArr, 0, countsBytes.Length);

            // Per-column importance averaged across the nmat axis.
            var imp = new float[nEmbd];
            for (int col = 0; col < nEmbd; col++)
            {
                double acc = 0;
                int count = 0;
                for (int mat = 0; mat < countsCount; mat++)
                {
                    float c = countsArr[mat];
                    if (c <= 0) continue;
                    acc += sumsArr[mat * nEmbd + col] / c;
                    count++;
                }
                imp[col] = count > 0 ? (float)(acc / count) : 0f;
            }
            byBaseName[baseName] = imp;
        }
        return byBaseName;
    }

    private static byte[] ReadRawBytes(LlamaGgufFile f, LlamaGgufTensorInfo t)
    {
        using var fs = File.OpenRead(f.SourcePath);
        fs.Seek(f.DataSectionFileOffset + t.ByteOffsetInDataSection, SeekOrigin.Begin);
        var bytes = new byte[t.ByteSize];
        ReadExactly(fs, bytes);
        return bytes;
    }

    private static void ReadExactly(Stream s, byte[] buffer)
    {
        int got = 0;
        while (got < buffer.Length)
        {
            int n = s.Read(buffer, got, buffer.Length - got);
            if (n <= 0) throw new EndOfStreamException("Truncated tensor read.");
            got += n;
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void GgmlToFloatDelegate(IntPtr src, IntPtr dst, long k);
}
