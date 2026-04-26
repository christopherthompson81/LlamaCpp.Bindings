namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Round-trip tests for <see cref="LlamaGgufSharding"/>. Build a small
/// synthetic GGUF (so we don't depend on a downloaded model for the
/// fast path), split it, merge, compare. The real-model check rides
/// on the existing <see cref="TestModelProvider"/> cache and
/// additionally verifies the merged result loads via
/// <see cref="LlamaModel"/>.
/// </summary>
public class GgufShardingTests
{
    [Fact]
    public async Task Path_Format_Matches_Upstream_Convention()
    {
        Assert.Equal("/m/llama-00002-of-00004.gguf",
            LlamaGgufSharding.FormatShardPath("/m/llama", 2, 4));
        Assert.Equal("model-00001-of-00001.gguf",
            LlamaGgufSharding.FormatShardPath("model", 1, 1));
        await Task.CompletedTask;
    }

    [Fact]
    public async Task Split_Then_Merge_Round_Trips_Synthetic_Gguf()
    {
        var dir = MakeTempDir();
        try
        {
            // Build an 8-tensor synthetic GGUF.
            var srcPath = Path.Combine(dir, "src.gguf");
            var w = new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test")
                .SetMetadata("test.dim", 16u);
            for (int i = 0; i < 8; i++)
            {
                var data = new float[16];
                for (int j = 0; j < data.Length; j++) data[j] = i * 100 + j;
                w.AddTensorF32($"t{i}", new long[] { 16 }, data);
            }
            await w.WriteAsync(srcPath);

            // Split by max-tensors=3 → 3 shards (sizes 3, 3, 2).
            var prefix = Path.Combine(dir, "shard");
            var splitResult = await LlamaGgufSharding.SplitAsync(
                srcPath, prefix, new LlamaGgufSplitOptions { MaxTensorsPerShard = 3 });

            Assert.Equal(3, splitResult.ShardCount);
            Assert.Equal(3, splitResult.ShardPaths.Count);
            foreach (var p in splitResult.ShardPaths) Assert.True(File.Exists(p));

            // Each shard should have the split.* metadata; only shard 0
            // should also carry the source's general.architecture etc.
            for (int i = 0; i < 3; i++)
            {
                var shard = LlamaGgufFile.Open(splitResult.ShardPaths[i]);
                var byKey = shard.Metadata.ToDictionary(m => m.Key, m => m.Value);

                Assert.Equal((ushort)i,         byKey["split.no"].AsUInt16());
                Assert.Equal((ushort)3,         byKey["split.count"].AsUInt16());
                Assert.Equal(8,                 byKey["split.tensors.count"].AsInt32());

                // Only shard 0 carries the source metadata.
                if (i == 0)
                {
                    Assert.True(byKey.ContainsKey("general.architecture"));
                    Assert.True(byKey.ContainsKey("test.dim"));
                }
                else
                {
                    Assert.False(byKey.ContainsKey("general.architecture"));
                    Assert.False(byKey.ContainsKey("test.dim"));
                }

                // Tensor count: 3 + 3 + 2.
                int expectedTensors = i switch { 0 => 3, 1 => 3, 2 => 2, _ => 0 };
                Assert.Equal(expectedTensors, shard.Tensors.Count);
            }

            // Merge back and check parity.
            var mergedPath = Path.Combine(dir, "merged.gguf");
            var mergeResult = await LlamaGgufSharding.MergeAsync(
                splitResult.ShardPaths[0], mergedPath);

            Assert.Equal(3, mergeResult.ShardCount);
            Assert.True(File.Exists(mergedPath));

            var merged = LlamaGgufFile.Open(mergedPath);
            // 8 tensors back, in original order.
            Assert.Equal(8, merged.Tensors.Count);
            for (int i = 0; i < 8; i++) Assert.Equal($"t{i}", merged.Tensors[i].Name);

            // Metadata: original keys preserved, no split.* leakage.
            var mergedByKey = merged.Metadata.ToDictionary(m => m.Key, m => m.Value);
            Assert.Equal("test", mergedByKey["general.architecture"].AsString());
            Assert.Equal(16u,    mergedByKey["test.dim"].AsUInt32());
            Assert.False(mergedByKey.ContainsKey("split.no"));
            Assert.False(mergedByKey.ContainsKey("split.count"));
            Assert.False(mergedByKey.ContainsKey("split.tensors.count"));

            // Tensor data round-trip: read each tensor's bytes from the
            // merged file at its declared offset and compare to original.
            var srcF = LlamaGgufFile.Open(srcPath);
            var srcBytes = await File.ReadAllBytesAsync(srcPath, TestContext.Current.CancellationToken);
            var mergedBytes = await File.ReadAllBytesAsync(mergedPath, TestContext.Current.CancellationToken);
            for (int i = 0; i < 8; i++)
            {
                var srcT = srcF.Tensors[i];
                var dstT = merged.Tensors[i];
                Assert.Equal(srcT.ByteSize, dstT.ByteSize);
                AssertBytesEqual(
                    srcBytes,    srcF.DataSectionFileOffset    + srcT.ByteOffsetInDataSection,
                    mergedBytes, merged.DataSectionFileOffset  + dstT.ByteOffsetInDataSection,
                    srcT.ByteSize);
            }
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Split_By_MaxBytes_Respects_Threshold()
    {
        var dir = MakeTempDir();
        try
        {
            // 4 tensors × 256 bytes = 1024 total; max=300 should give
            // shards of [256, 256, 256, 256] one-per-shard, since adding
            // a second 256-byte tensor would push past 300.
            var srcPath = Path.Combine(dir, "src.gguf");
            var w = new LlamaGgufWriter().SetMetadata("k", 1u);
            for (int i = 0; i < 4; i++)
            {
                w.AddTensorF32($"t{i}", new long[] { 64 }, new float[64]);
            }
            await w.WriteAsync(srcPath);

            var prefix = Path.Combine(dir, "shard");
            var result = await LlamaGgufSharding.SplitAsync(
                srcPath, prefix, new LlamaGgufSplitOptions { MaxBytesPerShard = 300 });

            Assert.Equal(4, result.ShardCount);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Split_Single_Tensor_Larger_Than_Max_Lands_Alone()
    {
        var dir = MakeTempDir();
        try
        {
            // One huge tensor (4 KB) and one tiny (4 B). MaxBytes=1024
            // would split them apart (huge can't fit any companion); the
            // huge one lands alone, the tiny in its own shard.
            var srcPath = Path.Combine(dir, "src.gguf");
            var big = new float[1024];   // 4 KB
            var small = new float[1];    // 4 B
            await new LlamaGgufWriter()
                .SetMetadata("k", 1u)
                .AddTensorF32("huge", new long[] { 1024 }, big)
                .AddTensorF32("tiny", new long[] { 1 }, small)
                .WriteAsync(srcPath);

            var prefix = Path.Combine(dir, "shard");
            var result = await LlamaGgufSharding.SplitAsync(
                srcPath, prefix, new LlamaGgufSplitOptions { MaxBytesPerShard = 1024 });

            // The first shard contains "huge" alone (would overflow if
            // we tried to add tiny). The second shard contains "tiny".
            Assert.Equal(2, result.ShardCount);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Merge_Detects_Inconsistent_Sibling_Set()
    {
        var dir = MakeTempDir();
        try
        {
            // Hand-write a "lone" file with split.count = 4 but no
            // siblings on disk; merge should fail with a clear error.
            var phantomPath = Path.Combine(dir, "phantom-00001-of-00004.gguf");
            await new LlamaGgufWriter()
                .SetMetadata("split.no", (ushort)0)
                .SetMetadata("split.count", (ushort)4)
                .SetMetadata("split.tensors.count", 0)
                .WriteAsync(phantomPath);

            var outPath = Path.Combine(dir, "merged.gguf");
            await Assert.ThrowsAnyAsync<FileNotFoundException>(() =>
                LlamaGgufSharding.MergeAsync(phantomPath, outPath));
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Split_Then_Merge_Real_Test_Model_Loads_Cleanly()
    {
        // Exercises the streamed-tensor path on a real model: split the
        // cached test GGUF into ~3 shards, merge back, then load the
        // merged result via LlamaModel. If the byte layout is wrong
        // anywhere, llama.cpp's loader catches it.
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var prefix = Path.Combine(dir, "shard");
            var splitResult = await LlamaGgufSharding.SplitAsync(
                modelPath, prefix, new LlamaGgufSplitOptions { MaxTensorsPerShard = 128 });

            // Real Qwen3-0.6B has 311 tensors; split-128 gives 3 shards.
            Assert.True(splitResult.ShardCount >= 2,
                $"Expected real model to produce multiple shards; got {splitResult.ShardCount}.");

            var mergedPath = Path.Combine(dir, "merged.gguf");
            await LlamaGgufSharding.MergeAsync(splitResult.ShardPaths[0], mergedPath);

            // The proof: load through the native model loader. A bad
            // byte offset or missing tensor would throw here.
            using var model = new LlamaModel(mergedPath, new LlamaModelParameters
            {
                GpuLayerCount = 0,
                UseMmap = true,
            });
            Assert.True(model.LayerCount > 0);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    // ----- helpers -----

    private static void AssertBytesEqual(byte[] a, long aOff, byte[] b, long bOff, long count)
    {
        for (long i = 0; i < count; i++)
        {
            Assert.Equal(a[aOff + i], b[bOff + i]);
        }
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-sharding-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
