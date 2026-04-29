using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>One per-tensor, per-candidate-type entry in a sensitivity sweep.</summary>
public sealed record LlamaQuantSensitivityScore(
    string TensorName,
    LlamaTensorType QuantType,
    double RawMse,
    double RelativeMse,
    long ElementCount,
    bool ImatrixWeighted,
    bool Skipped,
    string? SkipReason);

/// <summary>Final summary returned by <see cref="LlamaQuantSensitivity.MeasureAsync"/>.</summary>
public sealed record LlamaQuantSensitivityResult(
    string ModelPath,
    string? ImatrixPath,
    IReadOnlyList<LlamaTensorType> CandidateTypes,
    IReadOnlyList<LlamaQuantSensitivityScore> Scores,
    DateTime ComputedAtUtc,
    TimeSpan Elapsed);

/// <summary>Sub-step a sensitivity-sweep tensor is currently in.</summary>
public enum LlamaQuantSensitivityPhase
{
    /// <summary>Default; reported once when a new tensor starts.</summary>
    Tensor,
    /// <summary>Reading + dequantizing the source tensor to F32.</summary>
    SourceDequantize,
    /// <summary>Quantizing the F32 source to the candidate type.</summary>
    Quantize,
    /// <summary>Dequantizing the candidate's bytes back to F32.</summary>
    Dequantize,
    /// <summary>Computing MSE between source and round-trip.</summary>
    Score,
    /// <summary>One candidate finished; <c>CandidateRelativeMse</c> is populated.</summary>
    CandidateDone,
}

/// <summary>
/// Per-tensor / per-candidate progress reported during the sweep.
/// Fields beyond <see cref="CandidatesPerTensor"/> are populated when
/// <see cref="Phase"/> is per-candidate; consumers that don't care
/// about the lightboard can ignore them.
/// </summary>
public readonly record struct LlamaQuantSensitivityProgress(
    int TensorIndex,
    int TensorCount,
    string CurrentTensorName,
    int CandidatesPerTensor,
    int CandidateIndex = 0,
    LlamaTensorType? CandidateType = null,
    LlamaQuantSensitivityPhase Phase = LlamaQuantSensitivityPhase.Tensor,
    double? CandidateRelativeMse = null);

/// <summary>Knobs for <see cref="LlamaQuantSensitivity.MeasureAsync"/>.</summary>
public sealed class LlamaQuantSensitivityOptions
{
    /// <summary>Candidate types to score per tensor. <c>null</c> uses <see cref="LlamaQuantSensitivity.DefaultCandidateTypes"/>.</summary>
    public IReadOnlyList<LlamaTensorType>? CandidateTypes { get; set; }

    /// <summary>
    /// Optional path to an imatrix GGUF (output of
    /// <see cref="LlamaImatrix.ComputeAsync"/>). When provided, both
    /// the quantization and the score weighting use the per-column
    /// importance values; without it, both are unweighted.
    /// </summary>
    public string? ImatrixPath { get; set; }

    /// <summary>
    /// Skip 1-D tensors (norms, biases). Default <c>true</c> — they
    /// stay F32 in production quantization regardless of ftype, so
    /// scoring their sensitivity is moot.
    /// </summary>
    public bool SkipOneDimensional { get; set; } = true;

    /// <summary>
    /// Optional regex pattern. When set, only tensors whose names
    /// match (via <see cref="System.Text.RegularExpressions.Regex.IsMatch(string)"/>)
    /// are scored. Useful for fast iteration during development.
    /// </summary>
    public string? IncludeNameRegex { get; set; }

    /// <summary>
    /// Cap on candidates evaluated concurrently per tensor. Zero (the
    /// default) means "physical-core count" — each candidate is
    /// independent given the source F32 and runs on its own buffers,
    /// so this scales near-linearly with cores until memory bandwidth
    /// or RAM caps it. Set to 1 to force the old serial behavior.
    /// </summary>
    public int MaxDegreeOfParallelism { get; set; }
}

/// <summary>
/// Per-tensor quantization-sensitivity sweep. For each weight tensor T
/// in <paramref name="basePath"/>, dequantizes T to F32 (passthrough
/// if already F32), then for each candidate ftype Q rounds-trips
/// through <c>ggml_quantize_chunk</c> + the type's <c>to_float</c>
/// and reports both raw and relative reconstruction MSE. The score
/// is a function of (T, Q) alone, so the resulting table can be
/// re-thresholded any number of times without re-measuring.
/// </summary>
/// <remarks>
/// <para>
/// This is the "measurement" half of Adaptive Quantization. Phase 3
/// turns the score table into a per-tensor recipe by picking, for
/// each T, the lowest-precision Q whose relative MSE stays under a
/// configurable threshold.
/// </para>
/// <para>
/// The score uses ikawrakow's actual quantization kernels via
/// <c>ggml_quantize_chunk</c> — we don't re-implement the math, just
/// instrument it. With an imatrix the same column-weighting that the
/// production quantizer uses internally also weights our score, so a
/// "this Q is acceptable" decision tracks the same loss the
/// quantizer minimized.
/// </para>
/// </remarks>
public static class LlamaQuantSensitivity
{
    /// <summary>
    /// Default candidate ladder: from highest precision (F16) down
    /// through the K-quant family to the IQ family. Covers the range
    /// most users care about for "did this quant work."
    /// </summary>
    public static IReadOnlyList<LlamaTensorType> DefaultCandidateTypes { get; } = new[]
    {
        LlamaTensorType.F16,
        LlamaTensorType.BF16,
        LlamaTensorType.Q8_0,
        LlamaTensorType.Q6_K,
        LlamaTensorType.Q5_K,
        LlamaTensorType.Q4_K,
        LlamaTensorType.IQ4_XS,
        LlamaTensorType.Q3_K,
        LlamaTensorType.IQ3_S,
        LlamaTensorType.Q2_K,
        LlamaTensorType.IQ2_S,
    };

    public static Task<LlamaQuantSensitivityResult> MeasureAsync(
        string basePath,
        LlamaQuantSensitivityOptions? options = null,
        IProgress<LlamaQuantSensitivityProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(basePath);
        var opts = options ?? new LlamaQuantSensitivityOptions();
        return Task.Run(() => Measure(basePath, opts, progress, cancellationToken), cancellationToken);
    }

    private static LlamaQuantSensitivityResult Measure(
        string basePath,
        LlamaQuantSensitivityOptions opts,
        IProgress<LlamaQuantSensitivityProgress>? progress,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        LlamaBackend.EnsureInitialized();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var baseFile = LlamaGgufFile.Open(basePath);

        var candidates = opts.CandidateTypes ?? DefaultCandidateTypes;
        if (candidates.Count == 0)
        {
            throw new ArgumentException("No candidate types supplied.", nameof(opts));
        }

        // Imatrix loading (optional).
        Dictionary<string, float[]>? imatrixByTensor = null;
        if (!string.IsNullOrEmpty(opts.ImatrixPath))
        {
            imatrixByTensor = LoadImatrix(opts.ImatrixPath!);
        }

        // Filter the tensor list once so the progress denominator is honest.
        var includeRegex = !string.IsNullOrEmpty(opts.IncludeNameRegex)
            ? new System.Text.RegularExpressions.Regex(opts.IncludeNameRegex!)
            : null;
        var targets = baseFile.Tensors
            .Where(t => !opts.SkipOneDimensional || t.Dimensions.Length > 1)
            .Where(t => includeRegex is null || includeRegex.IsMatch(t.Name))
            .ToList();

        var scores = new List<LlamaQuantSensitivityScore>(targets.Count * candidates.Count);
        int parallelism = opts.MaxDegreeOfParallelism > 0
            ? opts.MaxDegreeOfParallelism
            : Math.Max(1, Environment.ProcessorCount / 2);

        for (int i = 0; i < targets.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var t = targets[i];
            int tensorIndexCapture = i + 1;
            // "New tensor" report — clears the lightboard for the UI.
            progress?.Report(new LlamaQuantSensitivityProgress(
                TensorIndex: tensorIndexCapture, TensorCount: targets.Count,
                CurrentTensorName: t.Name, CandidatesPerTensor: candidates.Count,
                Phase: LlamaQuantSensitivityPhase.Tensor));

            // Dequantize source to F32 once per tensor (cached across all candidates).
            progress?.Report(new LlamaQuantSensitivityProgress(
                TensorIndex: tensorIndexCapture, TensorCount: targets.Count,
                CurrentTensorName: t.Name, CandidatesPerTensor: candidates.Count,
                Phase: LlamaQuantSensitivityPhase.SourceDequantize));
            var source = ReadTensorAsFloat32(baseFile, t);
            float[]? imat = imatrixByTensor is not null && imatrixByTensor.TryGetValue(t.Name, out var col)
                ? col : null;

            // ||W||² for relative MSE — denominator. Computed once per tensor.
            double sumSq = 0;
            for (int j = 0; j < source.Length; j++) sumSq += (double)source[j] * source[j];
            double meanSq = source.Length > 0 ? sumSq / source.Length : 1;
            // Avoid /0 if a tensor is all-zeros; use 1 so the ratio at least
            // reports raw MSE rather than NaN.
            double denom = meanSq > 1e-30 ? meanSq : 1;

            int nPerRow = (int)t.Dimensions[0];
            long elements = source.Length;
            long nRows = elements / nPerRow;
            if (nRows * nPerRow != elements)
            {
                scores.Add(SkipScore(t, candidates, "tensor element count not divisible by ne[0]"));
                continue;
            }

            // Run candidates concurrently. Each candidate is independent
            // given the (now in-memory) source F32, so this scales with
            // cores until RAM bandwidth or memory caps it.
            var perCandidate = new LlamaQuantSensitivityScore?[candidates.Count];
            var po = new ParallelOptions
            {
                MaxDegreeOfParallelism = parallelism,
                CancellationToken      = ct,
            };
            try
            {
                Parallel.For(0, candidates.Count, po, ci =>
                {
                    ct.ThrowIfCancellationRequested();
                    var qtype = candidates[ci];
                    var ggmlQ = (ggml_type)(int)qtype;
                    var traitsPtr = NativeMethods.ggml_get_type_traits(ggmlQ);
                    if (traitsPtr == IntPtr.Zero)
                    {
                        perCandidate[ci] = new LlamaQuantSensitivityScore(
                            t.Name, qtype, double.NaN, double.NaN, elements,
                            imat is not null, Skipped: true,
                            SkipReason: "ggml_get_type_traits returned null");
                        return;
                    }
                    var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);

                    if (traits.blck_size > 1 && nPerRow % traits.blck_size != 0)
                    {
                        perCandidate[ci] = new LlamaQuantSensitivityScore(
                            t.Name, qtype, double.NaN, double.NaN, elements,
                            imat is not null, Skipped: true,
                            SkipReason: $"n_per_row {nPerRow} not divisible by block size {traits.blck_size}");
                        return;
                    }

                    var (rawMse, relMse) = ScoreOneWithProgress(
                        source, ggmlQ, traits, nRows, nPerRow, imat, denom,
                        progress, tensorIndexCapture, targets.Count, t.Name,
                        candidatesPerTensor: candidates.Count,
                        candidateIndex: ci, candidateType: qtype);

                    perCandidate[ci] = new LlamaQuantSensitivityScore(
                        t.Name, qtype, rawMse, relMse, elements,
                        ImatrixWeighted: imat is not null,
                        Skipped: false, SkipReason: null);

                    progress?.Report(new LlamaQuantSensitivityProgress(
                        TensorIndex: tensorIndexCapture, TensorCount: targets.Count,
                        CurrentTensorName: t.Name, CandidatesPerTensor: candidates.Count,
                        CandidateIndex: ci, CandidateType: qtype,
                        Phase: LlamaQuantSensitivityPhase.CandidateDone,
                        CandidateRelativeMse: relMse));
                });
            }
            catch (OperationCanceledException) { throw; }
            catch (AggregateException agg)
            {
                // Surface a clean cancellation if it was the cause; otherwise
                // rethrow the first inner exception so the message is useful.
                if (agg.InnerExceptions.Any(e => e is OperationCanceledException))
                    throw new OperationCanceledException(ct);
                throw agg.Flatten().InnerException ?? agg;
            }

            // Append results in candidate order so the score table is
            // deterministic regardless of completion order.
            foreach (var s in perCandidate)
                if (s is not null) scores.Add(s);
        }

        sw.Stop();
        return new LlamaQuantSensitivityResult(
            ModelPath: basePath,
            ImatrixPath: opts.ImatrixPath,
            CandidateTypes: candidates,
            Scores: scores,
            ComputedAtUtc: DateTime.UtcNow,
            Elapsed: sw.Elapsed);
    }

    /// <summary>
    /// Wraps <see cref="ScoreOne"/> with progress reports at the phase
    /// boundaries (quantize / dequantize / score). The lightboard relies
    /// on these to show "what's happening for the current candidate"
    /// even when a single tensor takes minutes.
    /// </summary>
    private static (double raw, double relative) ScoreOneWithProgress(
        float[] source,
        ggml_type qtype,
        ggml_type_traits traits,
        long nRows, int nPerRow,
        float[]? imatrix,
        double denom,
        IProgress<LlamaQuantSensitivityProgress>? progress,
        int tensorIndex, int tensorCount, string tensorName, int candidatesPerTensor,
        int candidateIndex, LlamaTensorType candidateType)
    {
        Report(LlamaQuantSensitivityPhase.Quantize);
        long quantBytes = nRows * (long)traits.type_size * (nPerRow / Math.Max(1L, traits.blck_size));
        if (traits.blck_size <= 0)
        {
            quantBytes = nRows * nPerRow * (long)traits.type_size;
        }
        var quantBuf = new byte[quantBytes];
        var roundTripped = new float[source.Length];

        unsafe
        {
            fixed (float* psrc = source)
            fixed (byte*  pdst = quantBuf)
            fixed (float* pimat = imatrix)
            {
                NativeMethods.ggml_quantize_chunk(
                    qtype, psrc, pdst,
                    start: 0, nrows: nRows, n_per_row: nPerRow,
                    imatrix: pimat);
            }

            Report(LlamaQuantSensitivityPhase.Dequantize);
            var toFloat = Marshal.GetDelegateForFunctionPointer<NativeMethods.GgmlToFloatDelegate>(traits.to_float);
            fixed (byte*  pdst = quantBuf)
            fixed (float* prt  = roundTripped)
            {
                toFloat((IntPtr)pdst, (IntPtr)prt, source.Length);
            }
        }

        Report(LlamaQuantSensitivityPhase.Score);
        double sumSqErr = 0;
        if (imatrix is not null)
        {
            for (int e = 0; e < source.Length; e++)
            {
                double diff = (double)source[e] - roundTripped[e];
                sumSqErr += imatrix[e % nPerRow] * diff * diff;
            }
        }
        else
        {
            for (int e = 0; e < source.Length; e++)
            {
                double diff = (double)source[e] - roundTripped[e];
                sumSqErr += diff * diff;
            }
        }
        double rawMse = source.Length > 0 ? sumSqErr / source.Length : 0;
        double relMse = rawMse / denom;
        return (rawMse, relMse);

        void Report(LlamaQuantSensitivityPhase phase) =>
            progress?.Report(new LlamaQuantSensitivityProgress(
                TensorIndex:        tensorIndex,
                TensorCount:        tensorCount,
                CurrentTensorName:  tensorName,
                CandidatesPerTensor: candidatesPerTensor,
                CandidateIndex:     candidateIndex,
                CandidateType:      candidateType,
                Phase:              phase));
    }

    /// <summary>
    /// Round-trip <paramref name="source"/> through quantize+dequantize
    /// for one candidate type and compute (raw MSE, relative MSE).
    /// </summary>
    private static unsafe (double raw, double relative) ScoreOne(
        float[] source,
        ggml_type qtype,
        ggml_type_traits traits,
        long nRows, int nPerRow,
        float[]? imatrix,
        double denom)
    {
        long quantBytes = nRows * (long)traits.type_size * (nPerRow / Math.Max(1L, traits.blck_size));
        // For non-blocked types (F16/BF16) blck_size==1 and the
        // formula simplifies to nRows * nPerRow * type_size.
        if (traits.blck_size <= 0)
        {
            quantBytes = nRows * nPerRow * (long)traits.type_size;
        }

        var quantBuf = new byte[quantBytes];
        var roundTripped = new float[source.Length];

        fixed (float* psrc = source)
        fixed (byte* pdst = quantBuf)
        fixed (float* pimat = imatrix)
        fixed (float* prt = roundTripped)
        {
            // Quantize. Pass nRows in one call — ggml_quantize_chunk
            // handles the full nrows × n_per_row buffer.
            NativeMethods.ggml_quantize_chunk(
                qtype, psrc, pdst,
                start: 0, nrows: nRows, n_per_row: nPerRow,
                imatrix: pimat);

            // Dequantize via the type-traits to_float function pointer.
            // Note: to_float is declared as void(*)(const void*, float*, int64_t)
            // — k is the total element count.
            var toFloat = Marshal.GetDelegateForFunctionPointer<NativeMethods.GgmlToFloatDelegate>(traits.to_float);
            toFloat((IntPtr)pdst, (IntPtr)prt, source.Length);
        }

        double sumSqErr = 0;
        if (imatrix is not null)
        {
            // Per-column weighting: rows × columns flat layout means
            // element index e maps to column e % nPerRow.
            for (int e = 0; e < source.Length; e++)
            {
                double diff = (double)source[e] - roundTripped[e];
                sumSqErr += imatrix[e % nPerRow] * diff * diff;
            }
        }
        else
        {
            for (int e = 0; e < source.Length; e++)
            {
                double diff = (double)source[e] - roundTripped[e];
                sumSqErr += diff * diff;
            }
        }

        double rawMse = source.Length > 0 ? sumSqErr / source.Length : 0;
        double relMse = rawMse / denom;
        return (rawMse, relMse);
    }

    private static LlamaQuantSensitivityScore SkipScore(
        LlamaGgufTensorInfo t, IReadOnlyList<LlamaTensorType> candidates, string reason)
    {
        // Single skip record covering all candidates so per-tensor
        // skip reporting stays one row in the output (not N rows of
        // duplicate info). Callers that flatten to a 2-D table can
        // expand if they need.
        return new LlamaQuantSensitivityScore(
            t.Name, candidates[0], double.NaN, double.NaN,
            ProductOf(t.Dimensions),
            ImatrixWeighted: false, Skipped: true, SkipReason: reason);
    }

    private static long ProductOf(long[] dims)
    {
        long p = 1; foreach (var d in dims) p *= d; return p;
    }

    /// <summary>
    /// Read a tensor's bytes and dequantize / convert to F32. Mirrors
    /// the helper in <see cref="LlamaLoraMerge"/> but extended to
    /// accept any quantized source via <c>to_float</c>, since a
    /// research user might want sensitivity scores against an already-
    /// quantized base.
    /// </summary>
    private static unsafe float[] ReadTensorAsFloat32(LlamaGgufFile file, LlamaGgufTensorInfo t)
    {
        long elements = ProductOf(t.Dimensions);
        if (elements > int.MaxValue)
        {
            throw new InvalidDataException(
                $"Tensor '{t.Name}' has {elements} elements, exceeding .NET array limit.");
        }
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
                var src = MemoryMarshal.Cast<byte, ushort>(raw.AsSpan());
                for (int i = 0; i < arr.Length; i++)
                    arr[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                break;
            }
            case 30: // BF16
            {
                var src = MemoryMarshal.Cast<byte, ushort>(raw.AsSpan());
                for (int i = 0; i < arr.Length; i++)
                {
                    uint hi = (uint)src[i] << 16;
                    arr[i] = BitConverter.UInt32BitsToSingle(hi);
                }
                break;
            }
            default:
            {
                // Quantized source — go through the type's to_float.
                var traitsPtr = NativeMethods.ggml_get_type_traits((ggml_type)t.TypeId);
                if (traitsPtr == IntPtr.Zero)
                {
                    throw new NotSupportedException(
                        $"Tensor '{t.Name}' has type {t.TypeId} with no type-traits — can't dequantize.");
                }
                var traits = Marshal.PtrToStructure<ggml_type_traits>(traitsPtr);
                var toFloat = Marshal.GetDelegateForFunctionPointer<NativeMethods.GgmlToFloatDelegate>(traits.to_float);
                fixed (byte* pq = raw)
                fixed (float* pa = arr)
                {
                    toFloat((IntPtr)pq, (IntPtr)pa, arr.Length);
                }
                break;
            }
        }
        return arr;
    }

    /// <summary>
    /// Load an imatrix GGUF and return per-tensor column-importance
    /// vectors. Format per <see cref="LlamaImatrix"/>: each tracked
    /// weight has <c>&lt;name&gt;.in_sum2</c> (squared-sum F32) and
    /// <c>&lt;name&gt;.counts</c> (sample count F32). Importance is
    /// <c>in_sum2 / counts</c>.
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
            // in_sum2 shape is [n_embd, nmat]; for dense tensors nmat=1.
            // We average across nmat for the per-column importance.
            int sumCount = sumsBytes.Length / sizeof(float);
            int countsCount = countsBytes.Length / sizeof(float);
            if (countsCount == 0) continue;

            int nEmbd = sumCount / countsCount;
            if (nEmbd * countsCount != sumCount) continue;

            var sumsArr = new float[sumCount];
            Buffer.BlockCopy(sumsBytes, 0, sumsArr, 0, sumsBytes.Length);
            var countsArr = new float[countsCount];
            Buffer.BlockCopy(countsBytes, 0, countsArr, 0, countsBytes.Length);

            var importance = new float[nEmbd];
            // Average importance across the `nmat` slabs (1 for dense,
            // n_experts for MUL_MAT_ID MoE — though our V1 imatrix only
            // collects dense matmuls).
            for (int m = 0; m < countsCount; m++)
            {
                if (countsArr[m] <= 0) continue;
                float inv = 1f / countsArr[m];
                for (int j = 0; j < nEmbd; j++)
                {
                    importance[j] += sumsArr[m * nEmbd + j] * inv / countsCount;
                }
            }
            byBaseName[baseName] = importance;
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

    private static void ReadExactly(Stream s, Span<byte> buf)
    {
        int got = 0;
        while (got < buf.Length)
        {
            int n = s.Read(buf[got..]);
            if (n <= 0) throw new EndOfStreamException("Truncated tensor read.");
            got += n;
        }
    }

    // ----- JSON save/load -----

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
        // Allow Infinity/-Infinity/NaN as quoted literals so a sweep
        // that produced an unusual rel-MSE value (catastrophic
        // round-trip on a degenerate tensor) doesn't lose its score
        // table. The recipe builder already handles NaN explicitly.
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    /// <summary>Serialize a sweep result to a JSON file. Pairs with <see cref="LoadFromJson"/>.</summary>
    public static void SaveToJson(LlamaQuantSensitivityResult result, string path)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentException.ThrowIfNullOrEmpty(path);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(result, JsonOpts));
    }

    /// <summary>Read a saved sweep result. Pairs with <see cref="SaveToJson"/>.</summary>
    public static LlamaQuantSensitivityResult LoadFromJson(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<LlamaQuantSensitivityResult>(json, JsonOpts)
            ?? throw new InvalidDataException($"Failed to deserialize sweep result from {path}.");
    }
}
