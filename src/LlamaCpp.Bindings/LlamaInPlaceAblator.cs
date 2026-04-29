using System.Buffers;

namespace LlamaCpp.Bindings;

/// <summary>
/// Runs successive sensitivity-profile ablations against a single
/// resident F16 <see cref="LlamaModel"/> by rewriting the affected
/// tensors' bytes in place between PPL passes — no per-cell GGUF
/// quantize, no per-cell model load.
/// </summary>
/// <remarks>
/// <para>
/// <strong>Why this exists.</strong> The disk-based campaign loop
/// (quantize a recipe to GGUF + load that GGUF + run PPL + delete)
/// pays roughly 22 s per cell on Qwen3-1.7B for the
/// model-load + CUDA-init + GPU-upload work. For a 22-cell campaign
/// at 4-way concurrency that's ~120 s of campaign wall time spent
/// on per-cell setup rather than scoring. Per the user observation
/// that "we're typically only swapping out one class of tensors or
/// even a single tensor," all that work is redundant — the F16
/// reference data for ~95 % of tensors is identical from cell to cell.
/// </para>
/// <para>
/// <strong>Numerical equivalence.</strong> We don't write a real Q-quant
/// tensor to GPU; we write F16-shaped bytes whose values were obtained
/// by F16 → F32 → quantize → F32 → F16. The dequantized values are
/// bit-identical to what a real Q-quant kernel would produce on the
/// fly during matmul. The matmul kernel itself differs (F16×input vs
/// Q-kernel×input), so PPL will agree with the disk-based path within
/// float32 reduction-order noise but is not strictly bit-identical.
/// See Run 24 in <c>docs/investigations/qwen3_qk_sensitivity.md</c>
/// for the verification pass.
/// </para>
/// <para>
/// <strong>Constraints:</strong>
/// <list type="bullet">
///   <item>The base model loaded into <see cref="Model"/> must be F16
///         (every quantizable tensor stored as F16). Other source types
///         allocate differently-sized GPU buffers and the round-trip
///         encoder would produce mismatched byte counts.</item>
///   <item>The source GGUF passed to the constructor must match the
///         model the <see cref="LlamaModel"/> was loaded from — the
///         ablator reads original F16 tensor bytes directly from the
///         file at the offsets recorded in the GGUF header.</item>
/// </list>
/// </para>
/// </remarks>
public sealed class LlamaInPlaceAblator : IDisposable
{
    private readonly LlamaGgufFile _sourceGguf;
    private readonly Dictionary<string, LlamaGgufTensorInfo> _tensorInfoByName;
    // Thread-safe across the parallel encode phase: workers read+write
    // concurrently when fetching original bytes for tensors they haven't
    // touched yet.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte[]> _originalDataCache
        = new(StringComparer.Ordinal);
    private readonly HashSet<string> _dirtyTensors = new(StringComparer.Ordinal);
    private bool _disposed;

    /// <summary>The persistent model whose tensors we rewrite between ablations.</summary>
    public LlamaModel Model { get; }

    /// <summary>Path to the F16 source GGUF the model was loaded from.</summary>
    public string SourceGgufPath => _sourceGguf.SourcePath;

    /// <param name="model">
    /// A loaded <see cref="LlamaModel"/>. The caller retains ownership
    /// and must not dispose it before disposing this ablator. The model
    /// is expected to be F16 (see remarks on
    /// <see cref="LlamaInPlaceAblator"/>).
    /// </param>
    /// <param name="sourceGgufPath">
    /// Path to the GGUF file <paramref name="model"/> was loaded from.
    /// Used to read original tensor bytes via the GGUF header offsets.
    /// </param>
    public LlamaInPlaceAblator(LlamaModel model, string sourceGgufPath)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrEmpty(sourceGgufPath);
        if (!File.Exists(sourceGgufPath))
            throw new FileNotFoundException($"Source GGUF not found: {sourceGgufPath}", sourceGgufPath);

        Model = model;
        _sourceGguf = LlamaGgufFile.Open(sourceGgufPath);
        _tensorInfoByName = _sourceGguf.Tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
    }

    /// <summary>
    /// Apply the listed ablations (overwriting any from a previous call
    /// that aren't repeated here), then run perplexity on
    /// <paramref name="corpus"/> against the now-ablated model.
    /// </summary>
    /// <param name="ablations">
    /// (tensorName, candidateType) pairs to apply. Each tensor is
    /// round-tripped through the candidate type's quantization and the
    /// resulting F16-encoded bytes are uploaded to the model's tensor
    /// buffer. Pass an empty list to score the unmodified F16 baseline.
    /// </param>
    /// <param name="corpus">UTF-8 corpus text for the PPL pass.</param>
    /// <param name="pplOptions">Knobs for <see cref="LlamaPerplexity.ComputeAsync"/>; pass null for defaults.</param>
    /// <param name="imatrixByTensor">
    /// Optional per-tensor importance matrices for imatrix-aware
    /// quantization. Keys are tensor names; values are column-importance
    /// vectors of length <c>Dimensions[0]</c>. Tensors not in the map
    /// are quantized unweighted.
    /// </param>
    /// <param name="cancellationToken">Cancellation observed during PPL.</param>
    public async Task<LlamaPerplexityResult> RunAblationAsync(
        IReadOnlyList<(string TensorName, LlamaTensorType Type)> ablations,
        string corpus,
        LlamaPerplexityOptions? pplOptions = null,
        IReadOnlyDictionary<string, float[]>? imatrixByTensor = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(ablations);
        ArgumentException.ThrowIfNullOrEmpty(corpus);
        ObjectDisposedException.ThrowIf(_disposed, this);

        var newAblationSet = new HashSet<string>(
            ablations.Select(a => a.TensorName), StringComparer.Ordinal);

        // Per-phase timing instrumentation (ABLATOR_TRACE=1). Lets us
        // see per-worker, per-phase wall times with timestamps so we
        // can identify CPU/GPU contention windows in concurrent runs.
        bool trace = Environment.GetEnvironmentVariable("ABLATOR_TRACE") == "1";
        int tid = Environment.CurrentManagedThreadId;
        string Stamp() => DateTime.UtcNow.ToString("HH:mm:ss.fff");
        var swPhase = new System.Diagnostics.Stopwatch();
        int restoreCount = 0;
        double encodeSeconds = 0, uploadSeconds = 0;
        var swInner = new System.Diagnostics.Stopwatch();

        // Phase 1: restore previously-dirty tensors not in this set.
        swPhase.Restart();
        foreach (var dirty in _dirtyTensors.ToList())
        {
            if (newAblationSet.Contains(dirty)) continue;
            Model.SetTensorData(dirty, GetOriginalBytes(dirty));
            _dirtyTensors.Remove(dirty);
            restoreCount++;
        }
        var restoreSeconds = swPhase.Elapsed.TotalSeconds;
        if (trace)
            Console.Error.WriteLine($"[ABL T{tid,-3} {Stamp()}] restore({restoreCount}): {restoreSeconds:F2}s");

        // Phase 2: apply each ablation in parallel.
        //
        // Encode is CPU-bound, single-threaded per tensor, and the
        // critical path of a cell — the next PPL pass can't start
        // until apply finishes. Earlier traces showed encode degrading
        // 4-5x as the campaign progressed, which we attributed to CPU
        // contention with concurrent workers' PPL softmax (each worker's
        // ComputePplAsync uses ProcessorCount/pplConcurrency threads).
        //
        // Parallelizing encode within the worker turns the apply phase
        // into a short bursty surge that briefly preempts the persistent
        // softmax workers — exactly the priority shape we want, without
        // an explicit task queue / Thread.Priority layer.
        //
        // F16 ablations stay sequential since they're zero-CPU memcpys
        // and the SetTensorData call would just queue on the same
        // CUDA stream as the parallel block.
        swPhase.Restart();
        var nonF16 = new List<(string Name, LlamaTensorType Type)>(ablations.Count);
        foreach (var (name, type) in ablations)
        {
            if (type == LlamaTensorType.F16)
            {
                swInner.Restart();
                Model.SetTensorData(name, GetOriginalBytes(name));
                uploadSeconds += swInner.Elapsed.TotalSeconds;
                _dirtyTensors.Add(name);
            }
            else
            {
                nonF16.Add((name, type));
            }
        }

        if (nonF16.Count > 0)
        {
            // Parallel encode + upload. DOP capped at ProcessorCount; the
            // burst is short enough (~3-5s on Qwen3-1.7B) that briefly
            // saturating CPU is the right trade-off vs continuing to
            // contend with concurrent softmax workers for 30+s.
            int dop = Math.Max(1, Environment.ProcessorCount);
            object accumLock = new();
            Parallel.ForEach(
                nonF16,
                new ParallelOptions { MaxDegreeOfParallelism = dop, CancellationToken = cancellationToken },
                ablation =>
                {
                    var (name, type) = ablation;
                    var sourceBytes = GetOriginalBytes(name);
                    if (!_tensorInfoByName.TryGetValue(name, out var info))
                    {
                        throw new InvalidOperationException(
                            $"Tensor '{name}' not present in source GGUF.");
                    }
                    if (info.Dimensions.Length < 1)
                    {
                        throw new InvalidOperationException(
                            $"Tensor '{name}' has no dimensions — cannot ablate.");
                    }

                    int colCount = (int)info.Dimensions[0];
                    long elementCount = 1;
                    foreach (var d in info.Dimensions) elementCount *= d;
                    long rowCount = elementCount / colCount;

                    ReadOnlySpan<float> imatrix = default;
                    if (imatrixByTensor is not null
                        && imatrixByTensor.TryGetValue(name, out var imatrixVec))
                    {
                        imatrix = imatrixVec;
                    }

                    var encoded = ArrayPool<byte>.Shared.Rent(sourceBytes.Length);
                    try
                    {
                        var swLocal = System.Diagnostics.Stopwatch.StartNew();
                        LlamaTensorRoundTrip.EncodeInto(
                            sourceBytes, type, rowCount, colCount, imatrix,
                            encoded.AsSpan(0, sourceBytes.Length));
                        var encLocal = swLocal.Elapsed.TotalSeconds;

                        swLocal.Restart();
                        Model.SetTensorData(name, encoded.AsSpan(0, sourceBytes.Length));
                        var upLocal = swLocal.Elapsed.TotalSeconds;

                        lock (accumLock)
                        {
                            encodeSeconds += encLocal;
                            uploadSeconds += upLocal;
                            _dirtyTensors.Add(name);
                        }
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(encoded);
                    }
                });
        }
        if (trace)
            Console.Error.WriteLine(
                $"[ABL T{tid,-3} {Stamp()}] apply({ablations.Count}): encode={encodeSeconds:F2}s upload={uploadSeconds:F2}s wall={swPhase.Elapsed.TotalSeconds:F2}s");

        // Phase 3: PPL.
        cancellationToken.ThrowIfCancellationRequested();
        swPhase.Restart();
        var result = await LlamaPerplexity.ComputeAsync(
            Model, corpus, pplOptions, progress: null, cancellationToken)
            .ConfigureAwait(false);
        if (trace)
            Console.Error.WriteLine($"[ABL T{tid,-3} {Stamp()}] ppl: {swPhase.Elapsed.TotalSeconds:F2}s ppl={result.Perplexity:F4}");

        return result;
    }

    /// <summary>
    /// Restore every previously-ablated tensor from cached F16 source
    /// bytes. After this returns, the model is back at the F16
    /// baseline. Equivalent to <see cref="RunAblationAsync"/> with an
    /// empty ablation list, but skips the PPL pass.
    /// </summary>
    public void RestoreBaseline()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        foreach (var dirty in _dirtyTensors.ToList())
        {
            Model.SetTensorData(dirty, GetOriginalBytes(dirty));
            _dirtyTensors.Remove(dirty);
        }
    }

    /// <summary>
    /// Approximate memory footprint of cached F16 source bytes (across
    /// all tensors that have been ablated at least once). Useful for
    /// status logging.
    /// </summary>
    public long CachedSourceBytes
    {
        get
        {
            long total = 0;
            foreach (var (_, bytes) in _originalDataCache) total += bytes.Length;
            return total;
        }
    }

    /// <summary>
    /// Read original F16 tensor bytes from the source GGUF. Cached on
    /// first call so subsequent ablations of the same tensor reuse the
    /// in-memory copy.
    /// </summary>
    private byte[] GetOriginalBytes(string tensorName)
    {
        // ConcurrentDictionary.GetOrAdd's value factory is not protected
        // against duplicate concurrent invocation for the same key — two
        // threads racing on a fresh cache miss may both run the loader.
        // That's fine here: both produce identical bytes, the second
        // assignment is a no-op, and the wasted file read is one-shot.
        return _originalDataCache.GetOrAdd(tensorName, name =>
        {
            if (!_tensorInfoByName.TryGetValue(name, out var info))
            {
                throw new InvalidOperationException(
                    $"Tensor '{name}' not present in source GGUF.");
            }

            var bytes = new byte[info.ByteSize];
            using var fs = new FileStream(
                _sourceGguf.SourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            fs.Seek(_sourceGguf.DataSectionFileOffset + info.ByteOffsetInDataSection, SeekOrigin.Begin);
            int read = 0;
            while (read < bytes.Length)
            {
                int n = fs.Read(bytes, read, bytes.Length - read);
                if (n == 0)
                {
                    throw new EndOfStreamException(
                        $"Truncated read of tensor '{name}' from {_sourceGguf.SourcePath}: " +
                        $"got {read} of {bytes.Length} bytes.");
                }
                read += n;
            }
            return bytes;
        });
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _originalDataCache.Clear();
        _dirtyTensors.Clear();
        // _sourceGguf has no managed resources to dispose; LlamaGgufFile
        // is metadata-only after Open() returns.
    }
}
