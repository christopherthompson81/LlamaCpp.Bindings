using System.Globalization;

namespace LlamaCpp.Bindings;

/// <summary>Knobs for <see cref="LlamaGgufSharding.SplitAsync"/>.</summary>
public sealed class LlamaGgufSplitOptions
{
    /// <summary>
    /// Maximum tensors per shard. <c>null</c> means unlimited. Mutually
    /// exclusive with <see cref="MaxBytesPerShard"/> — exactly one must
    /// be set.
    /// </summary>
    public int? MaxTensorsPerShard { get; set; }

    /// <summary>
    /// Maximum bytes per shard's tensor-data block. <c>null</c> means
    /// unlimited. Mutually exclusive with <see cref="MaxTensorsPerShard"/>.
    /// </summary>
    public long? MaxBytesPerShard { get; set; }
}

/// <summary>Per-tensor progress reported during split/merge.</summary>
public readonly record struct LlamaGgufShardingProgress(
    int Index,
    int Count,
    string Phase,
    string CurrentTensorName);

/// <summary>Final summary returned by <see cref="LlamaGgufSharding.SplitAsync"/>.</summary>
public sealed record LlamaGgufSplitResult(
    int ShardCount,
    long TotalBytes,
    IReadOnlyList<string> ShardPaths,
    TimeSpan Elapsed);

/// <summary>Final summary returned by <see cref="LlamaGgufSharding.MergeAsync"/>.</summary>
public sealed record LlamaGgufMergeResult(
    int ShardCount,
    long TotalBytes,
    string OutputPath,
    TimeSpan Elapsed);

/// <summary>
/// Pure-C# split / merge for GGUFs, byte-compatible with upstream
/// <c>llama-gguf-split</c>'s output. Reuses
/// <see cref="LlamaGgufFile"/> for reading and
/// <see cref="LlamaGgufWriter.AddTensorFromFile"/> for streamed copies,
/// so multi-GB shards never sit in managed memory.
/// </summary>
/// <remarks>
/// <para>
/// Shard naming follows llama.cpp's
/// <c>llama_split_path</c> convention: a prefix path (no extension)
/// expands to <c>&lt;prefix&gt;-NNNNN-of-NNNNN.gguf</c> with a
/// 5-digit zero-padded shard number, 1-based in the filename and
/// 0-based in the <c>split.no</c> metadata.
/// </para>
/// <para>
/// Sharded GGUFs carry three metadata keys identifying their place in
/// a set: <c>split.no</c> (u16, 0-based), <c>split.count</c> (u16,
/// total shards), <c>split.tensors.count</c> (i32, total tensors
/// across the whole set). The first shard also carries the source's
/// full metadata; subsequent shards carry only the split.* keys plus
/// their tensors. Merge reverses both halves of that arrangement.
/// </para>
/// </remarks>
public static class LlamaGgufSharding
{
    private const string SplitNoKey            = "split.no";
    private const string SplitCountKey         = "split.count";
    private const string SplitTensorsCountKey  = "split.tensors.count";

    /// <summary>
    /// Split <paramref name="sourcePath"/> into shards under
    /// <paramref name="outputPathPrefix"/> (no extension). Returns the
    /// list of written shard paths in order.
    /// </summary>
    public static Task<LlamaGgufSplitResult> SplitAsync(
        string sourcePath,
        string outputPathPrefix,
        LlamaGgufSplitOptions options,
        IProgress<LlamaGgufShardingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ArgumentException.ThrowIfNullOrEmpty(outputPathPrefix);
        ArgumentNullException.ThrowIfNull(options);
        if (options.MaxTensorsPerShard.HasValue == options.MaxBytesPerShard.HasValue)
        {
            // Both null or both set: ambiguous. Match upstream gguf-split's "exactly one" rule.
            throw new ArgumentException(
                "Exactly one of MaxTensorsPerShard or MaxBytesPerShard must be set.", nameof(options));
        }
        if (options.MaxTensorsPerShard is < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxTensorsPerShard must be >= 1.");
        if (options.MaxBytesPerShard is < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxBytesPerShard must be >= 1.");

        return Task.Run(() => Split(sourcePath, outputPathPrefix, options, progress, cancellationToken),
            cancellationToken);
    }

    /// <summary>
    /// Merge a sharded GGUF set into a single file. The shard count is
    /// read from <paramref name="firstShardPath"/>'s <c>split.count</c>
    /// metadata; sibling shards are located by name pattern in the same
    /// directory.
    /// </summary>
    public static Task<LlamaGgufMergeResult> MergeAsync(
        string firstShardPath,
        string outputPath,
        IProgress<LlamaGgufShardingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(firstShardPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        return Task.Run(() => Merge(firstShardPath, outputPath, progress, cancellationToken),
            cancellationToken);
    }

    /// <summary>
    /// Format a shard path: <c>"&lt;prefix&gt;-NNNNN-of-NNNNN.gguf"</c>
    /// with 1-based shard number, 5-digit zero pad. Mirrors
    /// llama.cpp's <c>llama_split_path</c>.
    /// </summary>
    public static string FormatShardPath(string prefix, int shardNumberOneBased, int shardCount) =>
        $"{prefix}-{shardNumberOneBased.ToString("D5", CultureInfo.InvariantCulture)}" +
        $"-of-{shardCount.ToString("D5", CultureInfo.InvariantCulture)}.gguf";

    // ----- split -----

    private static LlamaGgufSplitResult Split(
        string sourcePath, string outputPathPrefix,
        LlamaGgufSplitOptions options,
        IProgress<LlamaGgufShardingProgress>? progress,
        CancellationToken ct)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var src = LlamaGgufFile.Open(sourcePath);

        // 1. Plan shard boundaries — assign each tensor to a shard
        //    index, greedy fit by either tensor count or byte count.
        var shardIndices = PlanShardAssignment(src.Tensors, options);
        int shardCount = shardIndices.Length == 0 ? 0 : shardIndices.Max() + 1;
        if (shardCount == 0)
        {
            throw new InvalidDataException(
                $"Source GGUF has no tensors — nothing to split: {sourcePath}");
        }
        if (shardCount > ushort.MaxValue)
        {
            // split.count is stored as u16 in upstream — cap there.
            throw new InvalidOperationException(
                $"Computed shard count {shardCount} exceeds u16 range; loosen the size/tensor limit.");
        }

        // 2. Emit each shard. The first shard inherits the full
        //    metadata block; the rest carry only the split.* keys.
        var shardPaths = new string[shardCount];
        long totalBytes = 0;
        for (int i = 0; i < shardCount; i++)
        {
            ct.ThrowIfCancellationRequested();
            var shardPath = FormatShardPath(outputPathPrefix, i + 1, shardCount);
            shardPaths[i] = shardPath;

            var writer = new LlamaGgufWriter(src.Alignment);
            if (i == 0)
            {
                // First shard: carry every source metadata KV through —
                // split-aware loaders read the rest of the metadata from
                // here. Skip any pre-existing split.* keys (would
                // happen if someone re-splits an already-split file).
                foreach (var kv in src.Metadata)
                {
                    if (IsSplitKey(kv.Key)) continue;
                    writer.SetMetadata(kv.Key, kv.Value);
                }
            }
            writer.SetMetadata(SplitNoKey, (ushort)i);
            writer.SetMetadata(SplitCountKey, (ushort)shardCount);
            writer.SetMetadata(SplitTensorsCountKey, src.Tensors.Count);

            int tensorIndex = 0;
            int tensorTotal = shardIndices.Count(idx => idx == i);
            for (int t = 0; t < src.Tensors.Count; t++)
            {
                if (shardIndices[t] != i) continue;
                ct.ThrowIfCancellationRequested();
                var tensor = src.Tensors[t];
                writer.AddTensorFromFile(
                    name: tensor.Name,
                    typeId: tensor.TypeId,
                    shape: tensor.Dimensions,
                    sourcePath: src.SourcePath,
                    sourceOffsetInFile: src.DataSectionFileOffset + tensor.ByteOffsetInDataSection,
                    byteSize: tensor.ByteSize);
                tensorIndex++;
                progress?.Report(new LlamaGgufShardingProgress(
                    Index: tensorIndex, Count: tensorTotal,
                    Phase: $"shard {i + 1}/{shardCount}",
                    CurrentTensorName: tensor.Name));
            }

            writer.WriteAsync(shardPath, ct).GetAwaiter().GetResult();
            totalBytes += new FileInfo(shardPath).Length;
        }

        sw.Stop();
        return new LlamaGgufSplitResult(
            ShardCount: shardCount,
            TotalBytes: totalBytes,
            ShardPaths: shardPaths,
            Elapsed: sw.Elapsed);
    }

    /// <summary>Assign each tensor to a shard index using greedy fit by the configured limit.</summary>
    private static int[] PlanShardAssignment(
        IReadOnlyList<LlamaGgufTensorInfo> tensors, LlamaGgufSplitOptions options)
    {
        var assignment = new int[tensors.Count];
        int currentShard = 0;
        int countInShard = 0;
        long bytesInShard = 0;

        for (int i = 0; i < tensors.Count; i++)
        {
            // Decide whether this tensor would push the current shard over the limit.
            bool wouldOverflow = false;
            if (options.MaxTensorsPerShard is int maxN && countInShard >= maxN)
            {
                wouldOverflow = true;
            }
            else if (options.MaxBytesPerShard is long maxB
                     && countInShard > 0
                     && bytesInShard + tensors[i].ByteSize > maxB)
            {
                // Don't split if the shard is empty — a single tensor
                // larger than maxB is still atomic; we let it land alone.
                wouldOverflow = true;
            }

            if (wouldOverflow)
            {
                currentShard++;
                countInShard = 0;
                bytesInShard = 0;
            }
            assignment[i] = currentShard;
            countInShard++;
            bytesInShard += tensors[i].ByteSize;
        }
        return assignment;
    }

    private static bool IsSplitKey(string key) =>
        key == SplitNoKey || key == SplitCountKey || key == SplitTensorsCountKey;

    // ----- merge -----

    private static LlamaGgufMergeResult Merge(
        string firstShardPath, string outputPath,
        IProgress<LlamaGgufShardingProgress>? progress, CancellationToken ct)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var first = LlamaGgufFile.Open(firstShardPath);

        var splitCountKv = first.Metadata.FirstOrDefault(m => m.Key == SplitCountKey);
        if (splitCountKv is null)
        {
            throw new InvalidDataException(
                $"'{firstShardPath}' is not a sharded GGUF (no '{SplitCountKey}' metadata key).");
        }
        int shardCount = splitCountKv.Value.AsUInt16();
        if (shardCount < 1)
        {
            throw new InvalidDataException($"'{firstShardPath}' reports split.count = {shardCount}.");
        }

        // Locate each shard. Upstream's path format is parameterized by
        // a prefix, so we reverse-engineer the prefix from the input
        // path's "-NNNNN-of-NNNNN.gguf" suffix.
        var prefix = ExtractShardPrefix(firstShardPath, shardCount);
        var shardPaths = new string[shardCount];
        for (int i = 0; i < shardCount; i++)
        {
            shardPaths[i] = FormatShardPath(prefix, i + 1, shardCount);
            if (!File.Exists(shardPaths[i]))
            {
                throw new FileNotFoundException(
                    $"Sibling shard {i + 1}/{shardCount} not found at expected path: {shardPaths[i]}",
                    shardPaths[i]);
            }
        }

        // Build the merged writer: first shard's metadata (minus split.*)
        // plus all shards' tensors in order.
        var writer = new LlamaGgufWriter(first.Alignment);
        foreach (var kv in first.Metadata)
        {
            if (IsSplitKey(kv.Key)) continue;
            writer.SetMetadata(kv.Key, kv.Value);
        }

        int totalTensors = 0;
        long inputBytes = 0;
        for (int i = 0; i < shardCount; i++)
        {
            ct.ThrowIfCancellationRequested();
            var shard = i == 0 ? first : LlamaGgufFile.Open(shardPaths[i]);
            inputBytes += new FileInfo(shardPaths[i]).Length;

            for (int t = 0; t < shard.Tensors.Count; t++)
            {
                ct.ThrowIfCancellationRequested();
                var tensor = shard.Tensors[t];
                writer.AddTensorFromFile(
                    name: tensor.Name,
                    typeId: tensor.TypeId,
                    shape: tensor.Dimensions,
                    sourcePath: shard.SourcePath,
                    sourceOffsetInFile: shard.DataSectionFileOffset + tensor.ByteOffsetInDataSection,
                    byteSize: tensor.ByteSize);
                totalTensors++;
                progress?.Report(new LlamaGgufShardingProgress(
                    Index: totalTensors, Count: -1,
                    Phase: $"shard {i + 1}/{shardCount}",
                    CurrentTensorName: tensor.Name));
            }
        }

        writer.WriteAsync(outputPath, ct).GetAwaiter().GetResult();
        sw.Stop();
        return new LlamaGgufMergeResult(
            ShardCount: shardCount,
            TotalBytes: inputBytes,
            OutputPath: outputPath,
            Elapsed: sw.Elapsed);
    }

    /// <summary>
    /// Reverse-engineer the path prefix from an existing shard path of
    /// the form <c>"prefix-NNNNN-of-NNNNN.gguf"</c>. We don't trust the
    /// "of" component blindly — we cross-check it against the
    /// <c>split.count</c> metadata to catch a renamed file pointing at
    /// the wrong sibling set.
    /// </summary>
    private static string ExtractShardPrefix(string firstShardPath, int expectedShardCount)
    {
        var fileName = Path.GetFileNameWithoutExtension(firstShardPath);
        // Suffix layout: "...-NNNNN-of-NNNNN" (12 chars after the prefix).
        const int SuffixLen = 1 + 5 + 4 + 5; // "-NNNNN" + "-of-" + "NNNNN"
        if (fileName.Length <= SuffixLen)
        {
            throw new InvalidDataException(
                $"Shard filename '{fileName}' is shorter than expected '<prefix>-NNNNN-of-NNNNN'.");
        }
        var stem = fileName[..^SuffixLen];
        var suffix = fileName[^SuffixLen..];
        // Sanity check the suffix shape so a mis-named file doesn't
        // produce a phantom prefix and silently mis-locate siblings.
        if (suffix[0] != '-' || suffix[6] != '-' || suffix[7] != 'o' || suffix[8] != 'f' || suffix[9] != '-')
        {
            throw new InvalidDataException(
                $"Shard filename '{fileName}' suffix '{suffix}' isn't of the form '-NNNNN-of-NNNNN'.");
        }
        if (!int.TryParse(suffix.AsSpan(10, 5), NumberStyles.Integer, CultureInfo.InvariantCulture, out int filenameCount)
            || filenameCount != expectedShardCount)
        {
            throw new InvalidDataException(
                $"Shard '{firstShardPath}' filename declares 'of-{filenameCount:D5}' but " +
                $"its split.count metadata is {expectedShardCount}; shard set is internally inconsistent.");
        }
        var dir = Path.GetDirectoryName(firstShardPath) ?? string.Empty;
        return Path.Combine(dir, stem);
    }
}
