using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Dimensionality-reduction method for collapsing a stack of per-token
/// activation diffs (across all prompt pairs and positions) into a
/// single n_embd-sized direction per layer.
/// </summary>
public enum LlamaControlVectorMethod
{
    /// <summary>
    /// Average over all rows, then normalize to unit length. Cheap and
    /// stable; gives a "centroid" direction. Matches <c>llama-cvector-generator
    /// --method mean</c>.
    /// </summary>
    Mean,

    /// <summary>
    /// Top right singular vector of the (rows × n_embd) diff matrix
    /// computed via power iteration. Produces a direction that captures
    /// the dominant axis of variation between positive and negative
    /// prompts. Matches <c>llama-cvector-generator --method pca</c>
    /// (which is its default).
    /// </summary>
    Pca,
}

/// <summary>Knobs for <see cref="LlamaControlVectorTrainer.ComputeAsync"/>.</summary>
public sealed class LlamaControlVectorOptions
{
    /// <summary>Dimensionality-reduction method. Default <see cref="LlamaControlVectorMethod.Pca"/> matches upstream.</summary>
    public LlamaControlVectorMethod Method { get; set; } = LlamaControlVectorMethod.Pca;

    /// <summary>
    /// Power-iteration count for <see cref="LlamaControlVectorMethod.Pca"/>.
    /// Default 1000. Diminishing returns after ~500 for typical n_embd
    /// (1024-4096) but cheap so we err on convergence.
    /// </summary>
    public int PcaIterations { get; set; } = 1000;

    /// <summary>CPU thread count for the inference context. <c>-1</c> = llama.cpp default.</summary>
    public int ThreadCount { get; set; } = -1;

    /// <summary>
    /// Optional architecture hint to write into the output GGUF
    /// (<c>controlvector.model_hint</c>). <c>null</c> reads it from the
    /// model's <c>general.architecture</c> metadata.
    /// </summary>
    public string? ModelHint { get; set; }
}

/// <summary>Per-prompt-pair progress reported during training.</summary>
public readonly record struct LlamaControlVectorProgress(
    int PromptPairIndex,
    int PromptPairCount,
    string Phase);

/// <summary>Final summary returned by <see cref="LlamaControlVectorTrainer.ComputeAsync"/>.</summary>
public sealed record LlamaControlVectorResult(
    int LayerCount,
    int PromptPairCount,
    int EmbeddingSize,
    LlamaControlVectorMethod Method,
    TimeSpan Elapsed);

/// <summary>
/// Trains a control vector — a per-layer steering direction — from
/// contrastive prompt pairs. Output is byte-compatible with
/// <c>llama-cvector-generator</c>'s GGUF format and round-trips through
/// <see cref="LlamaControlVector"/>.
/// </summary>
/// <remarks>
/// <para>
/// Algorithm: for each prompt pair, decode the positive and negative
/// prompts through the model and capture the residual stream
/// (<c>l_out-N</c> tensors) at every layer via the eval callback. Compute
/// per-layer (positive − negative) row-wise differences across all
/// token positions, drop rows that are essentially zero (the diff is
/// dominated by padding noise on those positions), and stack the
/// remaining rows from every pair into per-layer matrices. Reduce each
/// per-layer matrix to a single n_embd direction via either the mean of
/// all rows or the top right singular vector (PCA via power iteration).
/// Normalize and write as a GGUF with one <c>direction.&lt;layer&gt;</c>
/// tensor per layer.
/// </para>
/// <para>
/// V1 uses dense matrix capture from the residual stream, matching the
/// upstream tool's mechanism. The output is loadable via
/// <see cref="LlamaControlVector.LoadFromFile(string, float)"/> and
/// applicable via <see cref="LlamaContext.SetControlVector"/>.
/// </para>
/// </remarks>
public static class LlamaControlVectorTrainer
{
    /// <summary>Train a control vector from <paramref name="positivePrompts"/> / <paramref name="negativePrompts"/> and write it to <paramref name="outputPath"/>.</summary>
    public static Task<LlamaControlVectorResult> ComputeAsync(
        LlamaModel model,
        IReadOnlyList<string> positivePrompts,
        IReadOnlyList<string> negativePrompts,
        string outputPath,
        LlamaControlVectorOptions? options = null,
        IProgress<LlamaControlVectorProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(positivePrompts);
        ArgumentNullException.ThrowIfNull(negativePrompts);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        if (positivePrompts.Count == 0)
            throw new ArgumentException("Need at least one prompt pair.", nameof(positivePrompts));
        if (positivePrompts.Count != negativePrompts.Count)
            throw new ArgumentException(
                $"Positive ({positivePrompts.Count}) and negative ({negativePrompts.Count}) prompt counts must match.",
                nameof(negativePrompts));

        var opts = options ?? new LlamaControlVectorOptions();
        return Task.Run(() => Compute(model, positivePrompts, negativePrompts, outputPath, opts, progress, cancellationToken),
                        cancellationToken);
    }

    private static LlamaControlVectorResult Compute(
        LlamaModel model,
        IReadOnlyList<string> positive,
        IReadOnlyList<string> negative,
        string outputPath,
        LlamaControlVectorOptions opts,
        IProgress<LlamaControlVectorProgress>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        int nEmbd = model.EmbeddingSize;
        int nLayers = model.LayerCount;
        if (nLayers < 2)
        {
            throw new InvalidOperationException(
                $"Model has {nLayers} layers; control-vector training needs at least 2.");
        }

        // Pre-tokenize every pair, padding to equal length within the pair.
        // Upstream pads with the last token of " " (a single space) — same
        // here for byte-compat with the reference output.
        var pairs = new List<(int[] pos, int[] neg, int len)>();
        int padToken = ResolvePadToken(model);
        int maxLen = 0;
        for (int i = 0; i < positive.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var p = model.Vocab.Tokenize(positive[i], addSpecial: true, parseSpecial: true);
            var n = model.Vocab.Tokenize(negative[i], addSpecial: true, parseSpecial: true);
            if (p.Length == 0 || n.Length == 0)
            {
                throw new InvalidOperationException(
                    $"Prompt pair {i} tokenized to zero tokens (pos={p.Length}, neg={n.Length}).");
            }
            int len = Math.Max(p.Length, n.Length);
            p = PadToLength(p, len, padToken);
            n = PadToLength(n, len, padToken);
            pairs.Add((p, n, len));
            if (len > maxLen) maxLen = len;
        }

        // Pick a context size that fits the longest pair. Cap at the
        // model's training context.
        int trainingCtx = Math.Max(8, model.TrainingContextSize);
        int chunkSize = Math.Min(maxLen, trainingCtx);
        if (chunkSize < maxLen)
        {
            throw new InvalidOperationException(
                $"Longest prompt pair is {maxLen} tokens but the model's training context is {trainingCtx}. " +
                "Shorten the prompts or use a model with a larger context.");
        }

        // Collector behind a GCHandle so the unmanaged callback can find it.
        var collector = new ControlVectorCollector(nEmbd, nLayers);
        var gch = GCHandle.Alloc(collector);
        try
        {
            var ctxParams = new LlamaContextParameters
            {
                ContextSize          = (uint)chunkSize,
                LogicalBatchSize     = (uint)chunkSize,
                PhysicalBatchSize    = (uint)chunkSize,
                MaxSequenceCount     = 1,
                ThreadCount          = opts.ThreadCount,
                BatchThreadCount     = opts.ThreadCount,
                EvalCallback         = GetEvalCallbackPointer(),
                EvalCallbackUserData = GCHandle.ToIntPtr(gch),
            };

            using var context = new LlamaContext(model, ctxParams);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var batch = NativeMethods.llama_batch_init(chunkSize, embd: 0, n_seq_max: 1);
            try
            {
                for (int i = 0; i < pairs.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var (pos, neg, len) = pairs[i];
                    progress?.Report(new LlamaControlVectorProgress(i + 1, pairs.Count, "encoding"));

                    // Positive pass.
                    collector.BeginPair(len, isPositive: true);
                    context.ClearKvCache();
                    PopulateBatchAllLogits(ref batch, pos, len);
                    unsafe
                    {
                        var rc = NativeMethods.llama_decode(context.Handle.DangerousHandle, batch);
                        if (rc != 0)
                        {
                            throw new LlamaException(
                                "llama_decode", rc,
                                $"llama_decode returned {rc} on positive pass for pair {i + 1}/{pairs.Count}.");
                        }
                    }

                    // Negative pass.
                    collector.BeginPair(len, isPositive: false);
                    context.ClearKvCache();
                    PopulateBatchAllLogits(ref batch, neg, len);
                    unsafe
                    {
                        var rc = NativeMethods.llama_decode(context.Handle.DangerousHandle, batch);
                        if (rc != 0)
                        {
                            throw new LlamaException(
                                "llama_decode", rc,
                                $"llama_decode returned {rc} on negative pass for pair {i + 1}/{pairs.Count}.");
                        }
                    }

                    // Compute (pos - neg) row-wise per layer and append non-zero rows.
                    collector.FinishPair();
                }
            }
            finally
            {
                NativeMethods.llama_batch_free(batch);
            }

            // Reduce per-layer row stack to a single n_embd direction.
            // Upstream skips the final layer; we follow suit.
            int directionLayerCount = nLayers - 1;
            progress?.Report(new LlamaControlVectorProgress(pairs.Count, pairs.Count, "computing"));
            var directions = new float[directionLayerCount][];
            for (int il = 0; il < directionLayerCount; il++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                directions[il] = opts.Method switch
                {
                    LlamaControlVectorMethod.Mean => ComputeMeanDirection(collector.LayerRows(il), nEmbd),
                    LlamaControlVectorMethod.Pca  => ComputePcaDirection(collector.LayerRows(il), nEmbd, opts.PcaIterations, cancellationToken),
                    _ => throw new ArgumentOutOfRangeException(nameof(opts.Method))
                };
            }

            sw.Stop();

            // Write GGUF.
            var modelHint = opts.ModelHint
                ?? (model.Metadata.TryGetValue("general.architecture", out var arch) ? arch : "unknown");
            var writer = new LlamaGgufWriter()
                .SetMetadata("general.architecture", "controlvector")
                .SetMetadata("controlvector.model_hint", modelHint)
                .SetMetadata("controlvector.layer_count", directionLayerCount);

            for (int il = 0; il < directionLayerCount; il++)
            {
                writer.AddTensorF32(
                    name: $"direction.{il + 1}",
                    shape: new long[] { nEmbd },
                    data: directions[il]);
            }
            writer.WriteAsync(outputPath, cancellationToken).GetAwaiter().GetResult();

            return new LlamaControlVectorResult(
                LayerCount: directionLayerCount,
                PromptPairCount: pairs.Count,
                EmbeddingSize: nEmbd,
                Method: opts.Method,
                Elapsed: sw.Elapsed);
        }
        finally
        {
            gch.Free();
        }
    }

    private static int ResolvePadToken(LlamaModel model)
    {
        // Upstream pads with " ".back() — the last token of a tokenized
        // single space. We replicate that to stay byte-compatible.
        var tokens = model.Vocab.Tokenize(" ", addSpecial: false, parseSpecial: false);
        if (tokens.Length == 0)
        {
            // Fallback: BOS or EOS or 0. Should never trigger on real
            // tokenizers but defensive coding here is cheap.
            return model.Vocab.Bos ?? model.Vocab.Eos ?? 0;
        }
        return tokens[^1];
    }

    private static int[] PadToLength(int[] tokens, int targetLength, int padToken)
    {
        if (tokens.Length >= targetLength) return tokens;
        var padded = new int[targetLength];
        tokens.CopyTo(padded, 0);
        for (int i = tokens.Length; i < targetLength; i++) padded[i] = padToken;
        return padded;
    }

    private static unsafe void PopulateBatchAllLogits(
        ref llama_batch batch, int[] tokens, int count)
    {
        batch.n_tokens = count;
        var tokPtr   = (int*)batch.token;
        var posPtr   = (int*)batch.pos;
        var nSeqPtr  = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;
        // We need llama.cpp to compute residual streams at every position,
        // so request logits at every position too — that prevents any
        // optimization that would skip the per-position layer outputs.
        for (int i = 0; i < count; i++)
        {
            tokPtr[i]      = tokens[i];
            posPtr[i]      = i;
            nSeqPtr[i]     = 1;
            seqIdArr[i][0] = 0;
            logits[i]      = 1;
        }
    }

    /// <summary>Average all rows then unit-normalize. Returns a fresh n_embd-sized array.</summary>
    private static float[] ComputeMeanDirection(IReadOnlyList<float[]> rows, int nEmbd)
    {
        var dir = new float[nEmbd];
        if (rows.Count == 0) return dir;

        // Sum into a double accumulator to avoid float-precision loss.
        var acc = new double[nEmbd];
        foreach (var row in rows)
        {
            for (int j = 0; j < nEmbd; j++) acc[j] += row[j];
        }
        double inv = 1.0 / rows.Count;
        for (int j = 0; j < nEmbd; j++) dir[j] = (float)(acc[j] * inv);

        Normalize(dir);
        return dir;
    }

    /// <summary>
    /// Top right singular vector of <paramref name="rows"/> via power
    /// iteration on M^T M, where M is the (rows.Count × n_embd) matrix
    /// formed by stacking the rows. Equivalent to the leading principal
    /// component of the diff distribution.
    /// </summary>
    private static float[] ComputePcaDirection(
        IReadOnlyList<float[]> rows,
        int nEmbd,
        int iterations,
        CancellationToken cancellationToken)
    {
        var dir = new float[nEmbd];
        if (rows.Count == 0) return dir;

        // Initialize z to a deterministic non-zero direction so two runs
        // of the trainer on the same input produce the same vector.
        var z = new double[nEmbd];
        for (int j = 0; j < nEmbd; j++) z[j] = 1.0 / Math.Sqrt(nEmbd);

        // Workspace: we stage M·z (size rows.Count) and M^T·u (size n_embd).
        var Mz = new double[rows.Count];
        var MtU = new double[nEmbd];

        for (int iter = 0; iter < iterations; iter++)
        {
            if ((iter & 0x1F) == 0) cancellationToken.ThrowIfCancellationRequested();

            // u = M z, then normalize.
            for (int r = 0; r < rows.Count; r++)
            {
                double s = 0;
                var row = rows[r];
                for (int j = 0; j < nEmbd; j++) s += row[j] * z[j];
                Mz[r] = s;
            }
            double uNorm = 0;
            for (int r = 0; r < rows.Count; r++) uNorm += Mz[r] * Mz[r];
            uNorm = Math.Sqrt(uNorm);
            if (uNorm < 1e-30) break; // matrix is essentially zero
            double uInv = 1.0 / uNorm;
            for (int r = 0; r < rows.Count; r++) Mz[r] *= uInv;

            // v = M^T u, then normalize.
            Array.Clear(MtU, 0, nEmbd);
            for (int r = 0; r < rows.Count; r++)
            {
                var row = rows[r];
                double mr = Mz[r];
                for (int j = 0; j < nEmbd; j++) MtU[j] += row[j] * mr;
            }
            double vNorm = 0;
            for (int j = 0; j < nEmbd; j++) vNorm += MtU[j] * MtU[j];
            vNorm = Math.Sqrt(vNorm);
            if (vNorm < 1e-30) break;
            double vInv = 1.0 / vNorm;
            for (int j = 0; j < nEmbd; j++) z[j] = MtU[j] * vInv;
        }

        for (int j = 0; j < nEmbd; j++) dir[j] = (float)z[j];
        Normalize(dir);
        return dir;
    }

    private static void Normalize(float[] v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++) sum += (double)v[i] * v[i];
        double n = Math.Sqrt(sum);
        if (n < 1e-30) return; // already zero — leave as-is
        float inv = (float)(1.0 / n);
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }

    /// <summary>Cached function-pointer for the eval-callback trampoline.</summary>
    private static IntPtr _cachedCallbackPointer;
    private static IntPtr GetEvalCallbackPointer()
    {
        if (_cachedCallbackPointer == IntPtr.Zero)
        {
            unsafe
            {
                _cachedCallbackPointer = (IntPtr)(delegate* unmanaged[Cdecl]<IntPtr, byte, IntPtr, byte>)
                    &EvalCallbackTrampoline;
            }
        }
        return _cachedCallbackPointer;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
    private static unsafe byte EvalCallbackTrampoline(IntPtr tensorPtr, byte ask, IntPtr userData)
    {
        try
        {
            if (userData == IntPtr.Zero || tensorPtr == IntPtr.Zero) return 1;
            var collector = GCHandle.FromIntPtr(userData).Target as ControlVectorCollector;
            if (collector is null) return 1;

            ref var t = ref Unsafe.AsRef<ggml_tensor>((void*)tensorPtr);
            return (byte)(collector.OnTensor(ref t, tensorPtr, ask: ask != 0) ? 1 : 0);
        }
        catch
        {
            return 1;
        }
    }

    /// <summary>
    /// Per-tool collector. Holds two scratch buffers (positive, negative)
    /// for the current prompt pair plus a per-layer accumulator of
    /// non-zero diff rows that survives across pairs.
    /// </summary>
    private sealed class ControlVectorCollector
    {
        private readonly int _nEmbd;
        private readonly int _nLayers;
        private readonly object _gate = new();

        // Per-pair scratch: layer index → [n_embd × n_tokens] flat buffer.
        // Indexed row-major: row r, col c at [r * n_embd + c].
        private readonly Dictionary<int, float[]> _posLayer = new();
        private readonly Dictionary<int, float[]> _negLayer = new();

        // Per-layer accumulated non-zero diff rows across all pairs.
        // Each entry is a self-contained n_embd-sized row.
        private readonly List<float[]>[] _layerRows;

        private int _currentTokens;
        private bool _isPositive;

        public ControlVectorCollector(int nEmbd, int nLayers)
        {
            _nEmbd = nEmbd;
            _nLayers = nLayers;
            _layerRows = new List<float[]>[nLayers];
            for (int i = 0; i < nLayers; i++) _layerRows[i] = new List<float[]>();
        }

        public IReadOnlyList<float[]> LayerRows(int layerIndex) => _layerRows[layerIndex];

        public void BeginPair(int nTokens, bool isPositive)
        {
            lock (_gate)
            {
                _currentTokens = nTokens;
                _isPositive = isPositive;
                if (isPositive) _posLayer.Clear();
                else            _negLayer.Clear();
            }
        }

        public unsafe bool OnTensor(ref ggml_tensor t, IntPtr tensorPtr, bool ask)
        {
            // Capture residual-stream tensors named "l_out-N". Filter to
            // ne[1] == n_tokens so partial-position computations
            // (e.g. last-position-only output projections) are skipped.
            var name = ReadName(ref t);
            if (!name.StartsWith("l_out-", StringComparison.Ordinal)) return false;

            if (ask) return true;

            int layerIndex = ParseLayerIndex(name);
            if (layerIndex < 0 || layerIndex >= _nLayers) return true;
            if (t.type != ggml_type.GGML_TYPE_F32) return true;
            if (t.ne[0] != _nEmbd) return true;
            if (t.ne[1] != _currentTokens) return true;

            // Stage activation bytes into managed memory. Layout in C#:
            // we keep [n_embd, n_tokens] but lay it out as a single flat
            // buffer of (n_tokens × n_embd) so each row is contiguous and
            // cheap to extract during diff. ggml's data is stored
            // contiguously in dim-0-major (i.e. column 0 is contiguous);
            // ggml_backend_tensor_get gives us the raw bytes in that
            // layout. We re-arrange per-row below.
            long bytes = (long)t.ne[0] * t.ne[1] * sizeof(float);
            var dst = new float[_nEmbd * _currentTokens];

            bool isHost = t.buffer != IntPtr.Zero
                && NativeMethods.ggml_backend_buffer_is_host(t.buffer);
            fixed (float* dstPtr = dst)
            {
                if (isHost && t.data != IntPtr.Zero)
                {
                    Buffer.MemoryCopy((void*)t.data, dstPtr, bytes, bytes);
                }
                else
                {
                    NativeMethods.ggml_backend_tensor_get(tensorPtr, dstPtr, 0, (nuint)bytes);
                }
            }

            lock (_gate)
            {
                var bucket = _isPositive ? _posLayer : _negLayer;
                bucket[layerIndex] = dst;
            }
            return true;
        }

        public void FinishPair()
        {
            // Compute (pos - neg) per layer, walk row by row, drop near-zero
            // rows, and append survivors to the per-layer accumulator.
            //
            // Layout in dst: [n_embd, n_tokens] in column-major (ggml's
            // native order — col 0 is contiguous in memory). To extract
            // row r (token r), we take the slice
            // [r*n_embd ... (r+1)*n_embd) of every layer's buffer? No —
            // actually the layout is dim-0-major, meaning ne[0]=n_embd is
            // the contiguous axis. For tensor [n_embd, n_tokens] that
            // means: token 0 fills the first n_embd floats, token 1 the
            // next n_embd, etc. So row r (token r) IS the slice
            // [r*n_embd ... (r+1)*n_embd).
            lock (_gate)
            {
                foreach (var kvp in _posLayer)
                {
                    int layer = kvp.Key;
                    if (!_negLayer.TryGetValue(layer, out var negData)) continue;
                    var posData = kvp.Value;
                    if (posData.Length != negData.Length) continue;

                    int nTokens = posData.Length / _nEmbd;
                    for (int r = 0; r < nTokens; r++)
                    {
                        int rowStart = r * _nEmbd;
                        var row = new float[_nEmbd];
                        bool nonZero = false;
                        for (int j = 0; j < _nEmbd; j++)
                        {
                            float d = posData[rowStart + j] - negData[rowStart + j];
                            row[j] = d;
                            if (!nonZero && Math.Abs(d) > 1e-6f) nonZero = true;
                        }
                        if (nonZero) _layerRows[layer].Add(row);
                    }
                }
                _posLayer.Clear();
                _negLayer.Clear();
            }
        }

        private static unsafe string ReadName(ref ggml_tensor t)
        {
            fixed (byte* p = t.name)
            {
                int len = 0;
                while (len < 64 && p[len] != 0) len++;
                return Encoding.UTF8.GetString(p, len);
            }
        }

        /// <summary>
        /// Parse <c>l_out-N</c> → <c>N</c>. Returns -1 if the suffix isn't
        /// a clean integer (a tensor-name format change upstream would
        /// trip this — better to skip than mis-attribute).
        /// </summary>
        private static int ParseLayerIndex(string name)
        {
            const string prefix = "l_out-";
            if (!name.StartsWith(prefix, StringComparison.Ordinal)) return -1;
            return int.TryParse(name.AsSpan(prefix.Length), out var idx) ? idx : -1;
        }
    }
}
