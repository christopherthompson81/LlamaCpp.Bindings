using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Per-chunk progress reported during importance-matrix collection.
/// </summary>
public readonly record struct LlamaImatrixProgress(
    int ChunkIndex,
    int ChunkCount,
    int TensorsTracked);

/// <summary>
/// Final summary returned by <see cref="LlamaImatrix.ComputeAsync"/>.
/// </summary>
public sealed record LlamaImatrixResult(
    int ChunkCount,
    int TokensProcessed,
    int TensorsTracked,
    TimeSpan Elapsed);

/// <summary>
/// Knobs for <see cref="LlamaImatrix.ComputeAsync"/>.
/// </summary>
public sealed class LlamaImatrixOptions
{
    /// <summary>
    /// Per-chunk context length in tokens. <c>0</c> (default) uses
    /// <c>min(model.TrainingContextSize, 512)</c> — the standard
    /// <c>llama-imatrix</c> chunk size.
    /// </summary>
    public int ContextSize { get; set; } = 0;

    /// <summary>
    /// Include the final <c>output.weight</c> matmul in the imatrix.
    /// Default <c>false</c> matches <c>llama-imatrix --output-tensor</c>
    /// being off by default — most quants don't quantize that tensor and
    /// thus don't benefit from its imatrix entry.
    /// </summary>
    public bool ProcessOutput { get; set; } = false;

    /// <summary>CPU thread count for context creation. <c>-1</c> (default) inherits llama.cpp's default.</summary>
    public int ThreadCount { get; set; } = -1;

    /// <summary>Prepend BOS to the corpus before tokenizing. Matches <c>llama-imatrix</c>.</summary>
    public bool AddBeginningOfSequence { get; set; } = true;

    /// <summary>
    /// Optional list of dataset names to write into the output's
    /// <c>imatrix.datasets</c> metadata. Purely informational — readers
    /// use it to label where the imatrix came from. When <c>null</c> we
    /// write a single entry "in-memory corpus".
    /// </summary>
    public IReadOnlyList<string>? DatasetNames { get; set; }
}

/// <summary>
/// Generates an importance-matrix GGUF from a calibration corpus —
/// entirely in C# on top of llama.cpp's per-tensor eval-callback hook.
/// Output is byte-compatible with <c>llama-imatrix</c>'s GGUF format and
/// can be passed to <see cref="LlamaQuantizationParameters"/> via the
/// imatrix field on a future quantize call.
/// </summary>
/// <remarks>
/// <para>
/// Mechanism: we tokenize the corpus, slide a context-sized window over
/// it, and for every chunk run a forward pass with a managed eval
/// callback wired into <c>llama_context_params.cb_eval</c>. The callback
/// fires once per compute-graph node; for matrix multiplications on
/// per-layer weight tensors (and optionally <c>output.weight</c>) we read
/// the activation input <c>src[1]</c>, accumulate the column-wise squared
/// sums, and bump the per-tensor sample counter. After all chunks finish
/// we serialize the running totals as F32 tensors in the output GGUF
/// alongside the <c>imatrix.*</c> metadata keys readers expect.
/// </para>
/// <para>
/// V1 supports dense (<c>GGML_OP_MUL_MAT</c>) only. Mixture-of-experts
/// models route some matmuls through <c>GGML_OP_MUL_MAT_ID</c> with a
/// per-token expert selection — that path is recognized but skipped, and
/// the resulting imatrix won't cover MoE expert weights. Adding it is
/// straightforward (mirrors the upstream loop) but waits for a model
/// where it matters.
/// </para>
/// </remarks>
public static class LlamaImatrix
{
    /// <summary>Compute an imatrix from <paramref name="corpus"/> and write it to <paramref name="outputPath"/>.</summary>
    public static Task<LlamaImatrixResult> ComputeAsync(
        LlamaModel model,
        string corpus,
        string outputPath,
        LlamaImatrixOptions? options = null,
        IProgress<LlamaImatrixProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrEmpty(corpus);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        var opts = options ?? new LlamaImatrixOptions();

        return Task.Run(() => Compute(model, corpus, outputPath, opts, progress, cancellationToken), cancellationToken);
    }

    private static LlamaImatrixResult Compute(
        LlamaModel model,
        string corpus,
        string outputPath,
        LlamaImatrixOptions opts,
        IProgress<LlamaImatrixProgress>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        int[] tokens = model.Vocab.Tokenize(
            corpus,
            addSpecial: opts.AddBeginningOfSequence,
            parseSpecial: false);
        if (tokens.Length < 2)
        {
            throw new InvalidOperationException(
                $"Corpus tokenized to {tokens.Length} token(s); imatrix needs at least 2.");
        }

        int trainingCtx = Math.Max(8, model.TrainingContextSize);
        int chunkSize = opts.ContextSize > 0
            ? Math.Min(opts.ContextSize, trainingCtx)
            : Math.Min(512, trainingCtx);
        if (tokens.Length < chunkSize)
        {
            chunkSize = Math.Max(8, tokens.Length);
        }
        int chunkCount = tokens.Length / chunkSize;
        if (chunkCount == 0)
        {
            throw new InvalidOperationException(
                $"Corpus has {tokens.Length} tokens but chunk size is {chunkSize}; need at least one full chunk.");
        }

        // Allocate the collector and pin it through a GCHandle so the
        // unmanaged callback can find it via the user-data IntPtr.
        var collector = new ImatrixCollector(opts.ProcessOutput);
        var gch = GCHandle.Alloc(collector);
        try
        {
            // Build context params with the eval callback wired in. The
            // trampoline is a static [UnmanagedCallersOnly] method so no
            // managed delegate needs to stay rooted across the call.
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
            int tokensProcessed = 0;
            var batch = NativeMethods.llama_batch_init(chunkSize, embd: 0, n_seq_max: 1);
            try
            {
                for (int chunk = 0; chunk < chunkCount; chunk++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int start = chunk * chunkSize;
                    context.ClearKvCache();
                    PopulateBatchAllLogits(ref batch, tokens, start, chunkSize);

                    unsafe
                    {
                        var rc = NativeMethods.llama_decode(context.Handle.DangerousHandle, batch);
                        if (rc != 0)
                        {
                            throw new LlamaException(
                                "llama_decode", rc,
                                $"llama_decode returned {rc} on imatrix chunk {chunk}/{chunkCount}.");
                        }
                    }

                    tokensProcessed += chunkSize;
                    progress?.Report(new LlamaImatrixProgress(
                        ChunkIndex: chunk + 1,
                        ChunkCount: chunkCount,
                        TensorsTracked: collector.TensorCount));
                }
            }
            finally
            {
                NativeMethods.llama_batch_free(batch);
            }
            sw.Stop();

            // Serialize. Naming keys mirror llama-imatrix's GGUF format
            // (general.type, imatrix.datasets, imatrix.chunk_count,
            // imatrix.chunk_size). The two per-tensor outputs (in_sum2,
            // counts) are F32 with the dimensions readers expect.
            var datasets = opts.DatasetNames is { Count: > 0 } d
                ? d
                : new[] { "in-memory corpus" };

            var writer = new LlamaGgufWriter()
                .SetMetadata("general.type", "imatrix")
                .SetMetadataStringArray("imatrix.datasets", datasets)
                .SetMetadata("imatrix.chunk_count", (uint)chunkCount)
                .SetMetadata("imatrix.chunk_size",  (uint)chunkSize);

            collector.Save(writer);
            writer.WriteAsync(outputPath, cancellationToken).GetAwaiter().GetResult();

            return new LlamaImatrixResult(
                ChunkCount: chunkCount,
                TokensProcessed: tokensProcessed,
                TensorsTracked: collector.TensorCount,
                Elapsed: sw.Elapsed);
        }
        finally
        {
            gch.Free();
        }
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

    /// <summary>
    /// Static trampoline that the native scheduler calls per compute
    /// graph node. The signature matches <c>ggml_backend_sched_eval_callback</c>
    /// (returns bool, takes tensor pointer + ask flag + user data).
    /// </summary>
    /// <remarks>
    /// Marked <c>UnmanagedCallersOnly</c> so the JIT emits a stable
    /// reverse-PInvoke stub. Throws are swallowed — propagating an
    /// exception across the unmanaged boundary corrupts the host process.
    /// </remarks>
    [UnmanagedCallersOnly(CallConvs = new[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
    private static unsafe byte EvalCallbackTrampoline(IntPtr tensorPtr, byte ask, IntPtr userData)
    {
        try
        {
            if (userData == IntPtr.Zero || tensorPtr == IntPtr.Zero) return 1;
            var collector = GCHandle.FromIntPtr(userData).Target as ImatrixCollector;
            if (collector is null) return 1;

            ref var t = ref Unsafe.AsRef<ggml_tensor>((void*)tensorPtr);
            return (byte)(collector.OnTensor(ref t, ask: ask != 0) ? 1 : 0);
        }
        catch
        {
            // Returning true keeps the scheduler going; throwing would
            // unwind into native code and crash the process.
            return 1;
        }
    }

    private static unsafe void PopulateBatchAllLogits(
        ref llama_batch batch, int[] tokens, int offset, int count)
    {
        batch.n_tokens = count;
        var tokPtr   = (int*)batch.token;
        var posPtr   = (int*)batch.pos;
        var nSeqPtr  = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;
        for (int i = 0; i < count; i++)
        {
            tokPtr[i]      = tokens[offset + i];
            posPtr[i]      = i;
            nSeqPtr[i]     = 1;
            seqIdArr[i][0] = 0;
            // Enable logits for *every* token. The values are unused —
            // imatrix only cares about activations on the way through —
            // but if we mark only the last token (an obvious optimization
            // for "skip output projection on the others") the scheduler
            // also prunes the last layer's FFN computation down to that
            // one token's slice. The collector's "small-batch guard"
            // (src1.ne[1] >= 16) then rejects those calls, and the
            // resulting imatrix is silently missing the last block's
            // ffn_down/gate/up entries — verified deterministic on
            // Qwen3-0.6B and Llama-3.2-1B. Marking every token costs
            // one additional output-projection matmul per chunk
            // (negligible) and keeps every layer's FFN running at
            // chunk-batch width.
            logits[i] = 1;
        }
    }

    /// <summary>
    /// Per-tensor accumulator that lives behind the eval-callback
    /// trampoline. Holds a dictionary keyed on the source-0 weight name
    /// so multiple matmul invocations on the same weight (across chunks
    /// and across positions within a chunk) sum into the same entry.
    /// </summary>
    private sealed class ImatrixCollector
    {
        private readonly bool _processOutput;
        private readonly object _gate = new();
        private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
        // Reused across callbacks: holds row*row in float, then folded
        // into the entry's double accumulator. Single-buffer is OK since
        // the eval callback fires sequentially per ggml node.
        private float[] _sqBuffer = Array.Empty<float>();

        public ImatrixCollector(bool processOutput) { _processOutput = processOutput; }

        public int TensorCount
        {
            get { lock (_gate) return _entries.Count; }
        }

        public unsafe bool OnTensor(ref ggml_tensor t, bool ask)
        {
            // V1: dense matmul only. MoE indirection (MUL_MAT_ID) is
            // recognized but skipped — we'd need to read the per-token
            // expert ids tensor and accumulate per-expert, which the
            // ImatrixCollector wireup is structured for but we haven't
            // implemented yet.
            if (t.op != ggml_op.GGML_OP_MUL_MAT) return false;

            // src[0] is the weight; src[1] is the activation input.
            var src0Ptr = ReadSrcPointer(ref t, 0);
            var src1Ptr = ReadSrcPointer(ref t, 1);
            if (src0Ptr == IntPtr.Zero || src1Ptr == IntPtr.Zero) return false;

            ref var src0 = ref Unsafe.AsRef<ggml_tensor>((void*)src0Ptr);
            ref var src1 = ref Unsafe.AsRef<ggml_tensor>((void*)src1Ptr);

            // Match upstream filter: skip small batches and non-F32
            // activations. The "small batch" guard avoids polluting the
            // imatrix with tiny single-token decode steps; we run with
            // chunk-sized batches so we'll always be well above 16.
            if (src1.ne[1] < 16 || src1.type != ggml_type.GGML_TYPE_F32) return false;

            var name = ReadName(ref src0);
            var filtered = FilterTensorName(name);

            // Apply the same name filter as llama-imatrix: only "blk."
            // tensors plus, optionally, output.weight.
            bool tracked = filtered.StartsWith("blk.", StringComparison.Ordinal)
                || (_processOutput && filtered == "output.weight");
            if (!tracked) return false;

            // Two-phase callback: ask=true asks "do you want this?" and
            // we just return true; ask=false is the actual data delivery.
            if (ask) return true;

            // src1 layout for a dense matmul: [n_embd, n_tokens] (rank 2)
            // or [n_embd, n_tokens, n_batch, ...] for higher-rank
            // batches. Strides are in src1.nb[]. For V1 we handle the
            // 2-D case (which covers Qwen3-style decoder-only models).
            long nEmbd = src1.ne[0];
            long nTokens = src1.ne[1];
            if (nEmbd <= 0 || nTokens <= 0) return false;

            // Stage activation bytes into a host buffer if needed. On
            // CUDA/Vulkan/Metal builds the activation lives on the GPU
            // and we have to copy it back; on CPU builds we read the
            // pointer directly.
            bool isHost = src1.buffer != IntPtr.Zero
                && NativeMethods.ggml_backend_buffer_is_host(src1.buffer);

            long bytes = nEmbd * nTokens * sizeof(float);
            if (bytes <= 0) return false;

            // Copy out under a lock so concurrent callbacks (parallel
            // sequences) don't trample the staging buffer. The hot path
            // — squaring the row in float and accumulating into the
            // entry's double[] — was a scalar inner loop pre-Round-1
            // and dominated wall time at 80% of the imatrix run on
            // Qwen3-1.7B. Vectorizing the squaring (TensorPrimitives)
            // and the float→double accumulate (Vector256.WidenLower)
            // halved the math phase and gave a 1.68× overall speedup.
            lock (_gate)
            {
                var entry = GetOrAddEntry(filtered, (int)nEmbd);
                if (entry.Values.Length != nEmbd) return false;

                fixed (byte* stage = entry.StagingBuffer((int)bytes))
                {
                    if (isHost && src1.data != IntPtr.Zero)
                    {
                        Buffer.MemoryCopy((void*)src1.data, stage, bytes, bytes);
                    }
                    else
                    {
                        NativeMethods.ggml_backend_tensor_get(src1Ptr, stage, 0, (nuint)bytes);
                    }

                    var act = (float*)stage;
                    long rowStrideBytes = unchecked((long)src1.nb[1]);
                    int n = (int)nEmbd;
                    var sq = _sqBuffer.Length >= n ? _sqBuffer : (_sqBuffer = new float[n]);
                    var values = entry.Values;
                    for (long row = 0; row < nTokens; row++)
                    {
                        var rowPtr = (float*)((byte*)act + row * rowStrideBytes);
                        // Vectorized square via TensorPrimitives (AVX2 on this CPU).
                        var rowSpan = new ReadOnlySpan<float>(rowPtr, n);
                        var sqSpan = sq.AsSpan(0, n);
                        TensorPrimitives.Multiply(rowSpan, rowSpan, sqSpan);
                        // Vectorized float→double accumulate (Vector256 of doubles).
                        AccumulateFloatToDouble(sq, values, n);
                    }
                    entry.Count += (int)nTokens;
                }
            }
            return true;
        }

        public void Save(LlamaGgufWriter writer)
        {
            // Sort for deterministic on-disk order — readers iterate
            // tensor info sequentially, so a stable order makes diffs
            // between runs meaningful.
            string[] names;
            lock (_gate)
            {
                names = _entries.Keys.ToArray();
            }
            Array.Sort(names, StringComparer.Ordinal);

            foreach (var name in names)
            {
                Entry e;
                lock (_gate)
                {
                    if (!_entries.TryGetValue(name, out e!)) continue;
                }
                int n = e.Values.Length;
                var sums = new float[n];
                for (int i = 0; i < n; i++) sums[i] = (float)e.Values[i];

                // Match llama-imatrix's tensor naming + shape contract.
                // For dense (nmat=1): in_sum2 is [n_embd, 1], counts is [1, 1].
                writer.AddTensorF32($"{name}.in_sum2", new long[] { n, 1 }, sums);
                writer.AddTensorF32($"{name}.counts", new long[] { 1, 1 }, new[] { (float)e.Count });
            }
        }

        /// <summary>
        /// Add each <paramref name="src"/> float (already squared) into
        /// the corresponding double slot of <paramref name="dst"/>. Hot
        /// loop: 4 doubles per iteration via Vector256, scalar tail.
        /// Vector128.WidenLower keeps the float→double widening in SIMD
        /// lanes (single VCVTPS2PD on AVX2).
        /// </summary>
        private static unsafe void AccumulateFloatToDouble(float[] src, double[] dst, int n)
        {
            int j = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                fixed (float* ps = src)
                fixed (double* pd = dst)
                {
                    for (; j + 4 <= n; j += 4)
                    {
                        var floats4 = Vector128.LoadUnsafe(ref *(ps + j));
                        var doubles4 = Vector256.WidenLower(floats4.ToVector256());
                        var acc = Vector256.LoadUnsafe(ref *(pd + j));
                        (acc + doubles4).StoreUnsafe(ref *(pd + j));
                    }
                }
            }
            for (; j < n; j++) dst[j] += src[j];
        }

        private Entry GetOrAddEntry(string name, int nEmbd)
        {
            if (_entries.TryGetValue(name, out var e)) return e;
            e = new Entry { Values = new double[nEmbd], Count = 0 };
            _entries[name] = e;
            return e;
        }

        private static unsafe IntPtr ReadSrcPointer(ref ggml_tensor t, int index)
        {
            // ggml_tensor.src is `struct ggml_tensor * src[10]` — we
            // mirror it as fixed long[10] to keep struct sizing simple.
            // Read as IntPtr-sized via a typed pointer to avoid 32-bit
            // platform footguns (we only target 64-bit, so this is a
            // direct read).
            ref long slot = ref Unsafe.Add(ref t.src[0], index);
            return (IntPtr)slot;
        }

        private static unsafe string ReadName(ref ggml_tensor t)
        {
            // Name is a fixed 64-byte char buffer, NUL-terminated UTF-8.
            fixed (byte* p = t.name)
            {
                int len = 0;
                while (len < 64 && p[len] != 0) len++;
                return Encoding.UTF8.GetString(p, len);
            }
        }

        /// <summary>
        /// Mirror of <c>filter_tensor_name</c> in upstream imatrix.cpp:
        /// extract content between two <c>#</c> markers when present,
        /// otherwise return the name unchanged. The marker form appears
        /// when ggml has merged tensors across a multi-batch pipeline.
        /// </summary>
        private static string FilterTensorName(string name)
        {
            int p = name.IndexOf('#');
            if (p < 0) return name;
            int start = p + 1;
            int q = name.IndexOf('#', start);
            return q < 0 ? name[start..] : name[start..q];
        }

        private sealed class Entry
        {
            public double[] Values = Array.Empty<double>();
            public int Count;
            private byte[] _staging = Array.Empty<byte>();
            public byte[] StagingBuffer(int minBytes)
            {
                if (_staging.Length < minBytes) _staging = new byte[minBytes];
                return _staging;
            }
        }
    }
}
