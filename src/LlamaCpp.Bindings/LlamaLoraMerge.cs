using System.Buffers.Binary;
using System.Runtime.InteropServices;
using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings;

/// <summary>One LoRA adapter to apply during merge, with its source path and scale multiplier.</summary>
public sealed record LlamaLoraAdapterInput(string Path, float Scale = 1.0f);

/// <summary>Per-tensor progress reported during merge.</summary>
public readonly record struct LlamaLoraMergeProgress(
    int TensorIndex,
    int TensorCount,
    string CurrentTensorName,
    bool IsMerged);

/// <summary>Final summary returned by <see cref="LlamaLoraMerge.MergeAsync"/>.</summary>
public sealed record LlamaLoraMergeResult(
    int TensorsTotal,
    int TensorsMerged,
    int TensorsCopied,
    long OutputBytes,
    LlamaHfConvertOutputType OutputType,
    TimeSpan Elapsed);

/// <summary>Knobs for <see cref="LlamaLoraMerge.MergeAsync"/>.</summary>
public sealed class LlamaLoraMergeOptions
{
    /// <summary>Output tensor type for merged weights. Default <see cref="LlamaHfConvertOutputType.F16"/>, matching upstream <c>export-lora</c>.</summary>
    public LlamaHfConvertOutputType OutputType { get; set; } = LlamaHfConvertOutputType.F16;
}

/// <summary>
/// Pure-C# LoRA-into-base-model merger. Walks the base GGUF's
/// tensors, looks up each <c>&lt;name&gt;.lora_a</c> /
/// <c>&lt;name&gt;.lora_b</c> in the adapter(s), and emits a merged
/// GGUF where the LoRAed tensors carry the combined weights and all
/// other tensors stream through unchanged. Equivalent to upstream's
/// <c>llama-export-lora</c>.
/// </summary>
/// <remarks>
/// <para>
/// Merge formula per tensor:
/// <c>W_merged = W_base + scale × (lora_a @ lora_b)</c>
/// where <c>scale = adapter_scale × adapter.lora.alpha / rank</c>
/// and <c>rank = lora_b.ne[0]</c>. Layout in ggml convention: base
/// has <c>ne[0] = in_features</c>, <c>ne[1] = out_features</c>;
/// <c>lora_a</c> is <c>[in, rank]</c> and <c>lora_b</c> is
/// <c>[rank, out]</c>; the matmul produces a delta of the same
/// <c>[in, out]</c> shape as the base.
/// </para>
/// <para>
/// V1 restricts both base and adapter tensor types to F32 / F16 /
/// BF16 — the merge math runs in F32 and the output is cast to the
/// chosen type. Quantized base or adapter tensors throw with a
/// clear "convert to F16/F32 first" error; the conventional workflow
/// is merge → quantize, not the other way around. Tensors that the
/// adapter doesn't touch stream through via the GGUF writer's
/// <c>AddTensorFromFile</c> primitive without ever sitting in
/// managed memory.
/// </para>
/// </remarks>
public static class LlamaLoraMerge
{
    /// <summary>
    /// Apply <paramref name="adapters"/> to <paramref name="basePath"/>
    /// and write the merged GGUF to <paramref name="outputPath"/>.
    /// </summary>
    public static Task<LlamaLoraMergeResult> MergeAsync(
        string basePath,
        IReadOnlyList<LlamaLoraAdapterInput> adapters,
        string outputPath,
        LlamaLoraMergeOptions? options = null,
        IProgress<LlamaLoraMergeProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(basePath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        ArgumentNullException.ThrowIfNull(adapters);
        if (adapters.Count == 0)
            throw new ArgumentException("Need at least one LoRA adapter to merge.", nameof(adapters));

        var opts = options ?? new LlamaLoraMergeOptions();
        return Task.Run(() => Merge(basePath, adapters, outputPath, opts, progress, cancellationToken),
            cancellationToken);
    }

    private static LlamaLoraMergeResult Merge(
        string basePath,
        IReadOnlyList<LlamaLoraAdapterInput> adapterInputs,
        string outputPath,
        LlamaLoraMergeOptions opts,
        IProgress<LlamaLoraMergeProgress>? progress,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        var baseFile = LlamaGgufFile.Open(basePath);
        var adapters = adapterInputs.Select(a => new LoadedAdapter(a.Path, a.Scale)).ToArray();

        // Validate every adapter is a LoRA adapter for the same architecture as the base.
        foreach (var adapter in adapters)
        {
            adapter.Validate(baseFile);
        }

        var writer = new LlamaGgufWriter(baseFile.Alignment);

        // Carry over base metadata. Stamp general.file_type to match
        // the chosen output type so a downstream Quantize-aware
        // consumer sees a consistent value.
        foreach (var kv in baseFile.Metadata)
        {
            if (kv.Key == "general.file_type") continue;  // we'll set our own
            writer.SetMetadata(kv.Key, kv.Value);
        }
        writer.SetMetadata("general.file_type", (uint)opts.OutputType);

        int merged = 0;
        int copied = 0;
        for (int i = 0; i < baseFile.Tensors.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var t = baseFile.Tensors[i];

            // Does *every* adapter have both lora_a and lora_b for this tensor?
            // We skip the merge unless every adapter contributes — partial
            // coverage is a sign of a malformed adapter set and we'd rather
            // surface it than silently apply some adapters but not others.
            bool everyAdapterHasLora = adapters.All(a =>
                a.File.Tensors.Any(x => x.Name == t.Name + ".lora_a") &&
                a.File.Tensors.Any(x => x.Name == t.Name + ".lora_b"));
            bool noAdapterHasLora = adapters.All(a =>
                !a.File.Tensors.Any(x => x.Name == t.Name + ".lora_a") &&
                !a.File.Tensors.Any(x => x.Name == t.Name + ".lora_b"));

            if (!everyAdapterHasLora && !noAdapterHasLora)
            {
                throw new InvalidDataException(
                    $"Base tensor '{t.Name}': some adapters provide lora_a/lora_b and some don't. " +
                    "V1 requires either every adapter or no adapter to cover each base tensor.");
            }

            if (noAdapterHasLora)
            {
                // Streamed passthrough — exactly like the GGUF Editor's save
                // path, but we already have the source open via LlamaGgufFile.
                writer.AddTensorFromFile(
                    name: t.Name,
                    typeId: t.TypeId,
                    shape: t.Dimensions,
                    sourcePath: baseFile.SourcePath,
                    sourceOffsetInFile: baseFile.DataSectionFileOffset + t.ByteOffsetInDataSection,
                    byteSize: t.ByteSize);
                copied++;
                progress?.Report(new LlamaLoraMergeProgress(i + 1, baseFile.Tensors.Count, t.Name, IsMerged: false));
                continue;
            }

            // Merge path. ggml convention: base.ne[0] = in_features,
            // base.ne[1] = out_features for a Linear weight (column-major-ish).
            if (t.Dimensions.Length != 2)
            {
                throw new InvalidDataException(
                    $"LoRA-targeted base tensor '{t.Name}' has {t.Dimensions.Length} dims; " +
                    "the LoRA merge math is only defined for 2-D weight matrices.");
            }
            int inDim  = (int)t.Dimensions[0];
            int outDim = (int)t.Dimensions[1];

            var baseF32 = ReadTensorAsFloat32(baseFile, t);
            // baseF32 is now [in*out] in ggml flat order:
            //   element [in_idx, out_idx] at index in_idx + out_idx*in_dim.

            foreach (var adapter in adapters)
            {
                var loraA = adapter.File.Tensors.First(x => x.Name == t.Name + ".lora_a");
                var loraB = adapter.File.Tensors.First(x => x.Name == t.Name + ".lora_b");
                ValidateLoraShapes(t, loraA, loraB);

                int rank = (int)loraB.Dimensions[0];
                var loraAFlat = ReadTensorAsFloat32(adapter.File, loraA);   // [in*rank]
                var loraBFlat = ReadTensorAsFloat32(adapter.File, loraB);   // [rank*out]

                // scale = adapter_scale × alpha / rank, matching upstream
                // (when alpha is 0 — early-PEFT convention — fall back to
                // adapter_scale alone).
                float scale = adapter.Alpha == 0
                    ? adapter.Scale
                    : adapter.Scale * adapter.Alpha / rank;

                AccumulateLoraDelta(baseF32, inDim, outDim, loraAFlat, loraBFlat, rank, scale);
            }

            // Cast the merged F32 result to the chosen output type.
            var outBytes = LlamaHfTensorTransforms.Passthrough(
                FloatsToBytes(baseF32),
                LlamaSafetensorsDtype.F32,
                opts.OutputType);
            writer.AddTensor(t.Name, outBytes.outTypeId, t.Dimensions, outBytes.outBytes);
            merged++;
            progress?.Report(new LlamaLoraMergeProgress(i + 1, baseFile.Tensors.Count, t.Name, IsMerged: true));
        }

        writer.WriteAsync(outputPath, ct).GetAwaiter().GetResult();
        sw.Stop();

        return new LlamaLoraMergeResult(
            TensorsTotal:  baseFile.Tensors.Count,
            TensorsMerged: merged,
            TensorsCopied: copied,
            OutputBytes:   new FileInfo(outputPath).Length,
            OutputType:    opts.OutputType,
            Elapsed:       sw.Elapsed);
    }

    /// <summary>
    /// Decode a tensor's raw bytes to a flat F32 array. Errors out
    /// for quantized types — V1 supports F32/F16/BF16 only.
    /// </summary>
    private static float[] ReadTensorAsFloat32(LlamaGgufFile file, LlamaGgufTensorInfo t)
    {
        // Read the raw bytes from the data section. We seek-and-read
        // here rather than use AddTensorFromFile (which streams) since
        // we need the data in managed memory for the merge math.
        using var fs = File.OpenRead(file.SourcePath);
        fs.Seek(file.DataSectionFileOffset + t.ByteOffsetInDataSection, SeekOrigin.Begin);
        var buf = new byte[t.ByteSize];
        ReadExactly(fs, buf);

        long elements = 1;
        foreach (var d in t.Dimensions) elements *= d;
        if (elements > int.MaxValue)
        {
            throw new InvalidDataException(
                $"Tensor '{t.Name}' has {elements} elements, exceeding .NET array limit.");
        }
        var arr = new float[(int)elements];

        switch (t.TypeId)
        {
            case 0: // F32
                Buffer.BlockCopy(buf, 0, arr, 0, buf.Length);
                break;
            case 1: // F16
                {
                    var src = MemoryMarshal.Cast<byte, ushort>(buf.AsSpan());
                    for (int i = 0; i < arr.Length; i++)
                    {
                        arr[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                    }
                    break;
                }
            case 30: // BF16
                {
                    var src = MemoryMarshal.Cast<byte, ushort>(buf.AsSpan());
                    for (int i = 0; i < arr.Length; i++)
                    {
                        uint hi = (uint)src[i] << 16;
                        arr[i] = BitConverter.UInt32BitsToSingle(hi);
                    }
                    break;
                }
            default:
                throw new NotSupportedException(
                    $"Tensor '{t.Name}' has type id {t.TypeId} ({t.Type?.ToString() ?? "?"}); " +
                    "LoRA merge V1 supports F32/F16/BF16 only. Convert the base model with " +
                    "the GGUF Editor's source-format export, or run the merge BEFORE quantizing.");
        }
        return arr;
    }

    private static void ValidateLoraShapes(LlamaGgufTensorInfo baseTensor, LlamaGgufTensorInfo a, LlamaGgufTensorInfo b)
    {
        if (a.Dimensions.Length != 2 || b.Dimensions.Length != 2)
        {
            throw new InvalidDataException(
                $"LoRA tensors for '{baseTensor.Name}' must be 2-D; got " +
                $"lora_a={a.Dimensions.Length}D, lora_b={b.Dimensions.Length}D.");
        }
        long aIn = a.Dimensions[0], aRank = a.Dimensions[1];
        long bRank = b.Dimensions[0], bOut = b.Dimensions[1];
        if (aRank != bRank)
        {
            throw new InvalidDataException(
                $"LoRA rank mismatch on '{baseTensor.Name}': lora_a has rank {aRank}, lora_b has rank {bRank}.");
        }
        if (aIn != baseTensor.Dimensions[0])
        {
            throw new InvalidDataException(
                $"LoRA in-feature mismatch on '{baseTensor.Name}': base has {baseTensor.Dimensions[0]}, lora_a has {aIn}.");
        }
        if (bOut != baseTensor.Dimensions[1])
        {
            throw new InvalidDataException(
                $"LoRA out-feature mismatch on '{baseTensor.Name}': base has {baseTensor.Dimensions[1]}, lora_b has {bOut}.");
        }
    }

    /// <summary>
    /// Accumulate <c>baseFlat += scale × (lora_a @ lora_b)</c> in-place.
    /// </summary>
    /// <remarks>
    /// Loop order is (out, rank, in) so the inner sweep is contiguous
    /// over the in-axis of both base and lora_a — best cache behaviour
    /// without needing SIMD intrinsics. For typical hidden=4096,
    /// rank=8, that's ~16M ops per tensor; on modern CPUs the JIT
    /// unrolls the inner loop and reaches >1 GFLOP/s, so a 7B model's
    /// merge takes single-digit seconds.
    /// </remarks>
    private static void AccumulateLoraDelta(
        float[] baseFlat, int inDim, int outDim,
        float[] loraA, float[] loraB,
        int rank, float scale)
    {
        for (int o = 0; o < outDim; o++)
        {
            int baseOffset = o * inDim;
            int loraBBase  = o * rank;
            for (int r = 0; r < rank; r++)
            {
                float bScaled = loraB[loraBBase + r] * scale;
                int loraABase = r * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    baseFlat[baseOffset + i] += loraA[loraABase + i] * bScaled;
                }
            }
        }
    }

    private static byte[] FloatsToBytes(float[] arr)
    {
        var bytes = new byte[arr.Length * sizeof(float)];
        Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static void ReadExactly(Stream s, Span<byte> buf)
    {
        int got = 0;
        while (got < buf.Length)
        {
            int n = s.Read(buf[got..]);
            if (n <= 0) throw new EndOfStreamException("Truncated read while loading tensor data.");
            got += n;
        }
    }

    private sealed class LoadedAdapter
    {
        public string SourcePath { get; }
        public LlamaGgufFile File { get; }
        public float Alpha { get; }
        public float Scale { get; }

        public LoadedAdapter(string path, float scale)
        {
            SourcePath = path;
            File = LlamaGgufFile.Open(path);
            Scale = scale;
            // adapter.lora.alpha is the canonical key; we expose it as
            // float regardless of whether the file stored it as f32 or u32.
            var alphaKv = File.Metadata.FirstOrDefault(m => m.Key == "adapter.lora.alpha");
            Alpha = alphaKv?.Value.Type switch
            {
                LlamaGgufType.Float32 => alphaKv.Value.AsFloat32(),
                LlamaGgufType.Uint32  => alphaKv.Value.AsUInt32(),
                LlamaGgufType.Int32   => alphaKv.Value.AsInt32(),
                _ => 0f,
            };
        }

        public void Validate(LlamaGgufFile baseFile)
        {
            string Get(string key) => File.Metadata.FirstOrDefault(m => m.Key == key)?.Value.AsString() ?? "";
            string baseGet(string key) => baseFile.Metadata.FirstOrDefault(m => m.Key == key)?.Value.AsString() ?? "";

            var generalType = Get("general.type");
            if (generalType != "adapter")
            {
                throw new InvalidDataException(
                    $"Adapter '{SourcePath}' has general.type='{generalType}'; expected 'adapter'.");
            }
            var adapterType = Get("adapter.type");
            if (adapterType != "lora")
            {
                throw new InvalidDataException(
                    $"Adapter '{SourcePath}' has adapter.type='{adapterType}'; expected 'lora'.");
            }
            var adapterArch = Get("general.architecture");
            var baseArch    = baseGet("general.architecture");
            if (!string.Equals(adapterArch, baseArch, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Architecture mismatch: base is '{baseArch}', adapter '{SourcePath}' is '{adapterArch}'.");
            }
        }
    }
}
