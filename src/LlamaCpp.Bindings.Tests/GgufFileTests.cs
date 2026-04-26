namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Round-trip + mutation tests for <see cref="LlamaGgufFile"/>. The pure-C#
/// reader is the editor's foundation: if it disagrees with the writer, or
/// if it can't load a real GGUF byte-for-byte, the editor is broken.
/// </summary>
public class GgufFileTests
{
    [Fact]
    public async Task Reader_Round_Trips_Every_Metadata_Type_Through_The_Writer()
    {
        // Write a synthetic GGUF that exercises every scalar + array type
        // we support, then read it back and verify each value survives.
        var dir = MakeTempDir();
        try
        {
            var path = Path.Combine(dir, "round-trip.gguf");
            await new LlamaGgufWriter()
                .SetMetadata("scalar.u8",  (byte)200)
                .SetMetadata("scalar.i8",  (sbyte)-7)
                .SetMetadata("scalar.u16", (ushort)40000)
                .SetMetadata("scalar.i16", (short)-12345)
                .SetMetadata("scalar.u32", 4_000_000_000u)
                .SetMetadata("scalar.i32", -123_456_789)
                .SetMetadata("scalar.u64", 1234567890123456789ul)
                .SetMetadata("scalar.i64", -1234567890123456789L)
                .SetMetadata("scalar.f32", 3.14159f)
                .SetMetadata("scalar.f64", Math.E)
                .SetMetadata("scalar.bool", true)
                .SetMetadata("scalar.str", "hello world")
                .SetMetadataStringArray("arr.str", new[] { "alpha", "beta", "γ" })
                .SetMetadata("arr.u32",
                    LlamaGgufValue.PrimitiveArray<uint>(new uint[] { 1u, 2u, 3u, 4u, 5u }))
                .SetMetadata("arr.f32",
                    LlamaGgufValue.PrimitiveArray<float>(new float[] { 0.5f, -0.25f, 1.5e30f }))
                .WriteAsync(path);

            var f = LlamaGgufFile.Open(path);
            Assert.Equal(LlamaGgufWriter.Version, f.Version);

            var byKey = f.Metadata.ToDictionary(m => m.Key, m => m.Value);
            Assert.Equal((byte)200,                       byKey["scalar.u8"].AsUInt8());
            Assert.Equal((sbyte)-7,                       byKey["scalar.i8"].AsInt8());
            Assert.Equal((ushort)40000,                   byKey["scalar.u16"].AsUInt16());
            Assert.Equal((short)-12345,                   byKey["scalar.i16"].AsInt16());
            Assert.Equal(4_000_000_000u,                  byKey["scalar.u32"].AsUInt32());
            Assert.Equal(-123_456_789,                    byKey["scalar.i32"].AsInt32());
            Assert.Equal(1234567890123456789ul,           byKey["scalar.u64"].AsUInt64());
            Assert.Equal(-1234567890123456789L,           byKey["scalar.i64"].AsInt64());
            Assert.Equal(3.14159f,                        byKey["scalar.f32"].AsFloat32());
            Assert.Equal(Math.E,                          byKey["scalar.f64"].AsFloat64(), 12);
            Assert.True(byKey["scalar.bool"].AsBool());
            Assert.Equal("hello world",                   byKey["scalar.str"].AsString());

            Assert.Equal(new[] { "alpha", "beta", "γ" },  byKey["arr.str"].AsStringArray());
            Assert.Equal(new uint[] { 1, 2, 3, 4, 5 },    byKey["arr.u32"].AsArray<uint>());
            Assert.Equal(new float[] { 0.5f, -0.25f, 1.5e30f }, byKey["arr.f32"].AsArray<float>());
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public void Reader_Loads_Real_Model_Without_Loading_Tensor_Data()
    {
        // Open the cached test model (Qwen3-0.6B). We expect the reader
        // to materialize every metadata KV and every tensor info entry,
        // and to NOT pull the tensor data into memory (managed allocation
        // should be tiny relative to the file size).
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var f = LlamaGgufFile.Open(modelPath);

        Assert.True(f.Metadata.Count > 10, $"Expected real model to have many metadata entries; got {f.Metadata.Count}.");
        Assert.True(f.Tensors.Count > 100, $"Expected real model to have many tensors; got {f.Tensors.Count}.");

        // general.architecture is one of the few KVs every llama.cpp model has.
        var arch = f.Metadata.FirstOrDefault(m => m.Key == "general.architecture");
        Assert.NotNull(arch);
        Assert.Equal(LlamaGgufType.String, arch!.Value.Type);

        // Every tensor should have at least one dim and a positive byte
        // size. If the size computation is wrong, an editor save would
        // produce a corrupted output, so this is a load-bearing check.
        foreach (var t in f.Tensors)
        {
            Assert.True(t.Dimensions.Length > 0, $"Tensor {t.Name} has 0 dims.");
            Assert.True(t.ByteSize > 0,          $"Tensor {t.Name} has 0 byte size.");
            Assert.True(t.ByteOffsetInDataSection >= 0);
        }

        // Data section should land within the file.
        var fileSize = new FileInfo(modelPath).Length;
        Assert.True(f.DataSectionFileOffset > 0 && f.DataSectionFileOffset < fileSize);
    }

    [Fact]
    public async Task SaveAs_Mutating_A_String_Value_Round_Trips_Through_LlamaModel_Loader()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "mutated.gguf");

            // Open, mutate, save.
            var f = LlamaGgufFile.Open(modelPath);
            var nameEntry = f.Metadata.First(m => m.Key == "general.architecture");
            var originalArch = nameEntry.Value.AsString();

            // Insert a fresh metadata entry plus mutate an existing one's value.
            f.Metadata.Add(new LlamaGgufMetadataEntry(
                "ggufeditor.test_marker", LlamaGgufValue.String("set-by-test")));
            // Edit general.name (every llama.cpp model has it).
            var nameKv = f.Metadata.FirstOrDefault(m => m.Key == "general.name");
            if (nameKv is not null)
            {
                nameKv.Value = LlamaGgufValue.String("Renamed by editor test");
            }

            await f.SaveAsAsync(outPath, TestContext.Current.CancellationToken);

            // Round-trip through the reader. Count may differ by 1 if the
            // source didn't have a general.alignment KV — the writer
            // auto-adds it for self-describing output. That's correct
            // behavior, so we assert >= rather than ==.
            var roundTripped = LlamaGgufFile.Open(outPath);
            Assert.True(roundTripped.Metadata.Count >= f.Metadata.Count,
                $"Round-tripped metadata count ({roundTripped.Metadata.Count}) shouldn't drop below source ({f.Metadata.Count}).");
            Assert.Equal(f.Tensors.Count,  roundTripped.Tensors.Count);

            var marker = roundTripped.Metadata.FirstOrDefault(m => m.Key == "ggufeditor.test_marker");
            Assert.NotNull(marker);
            Assert.Equal("set-by-test", marker!.Value.AsString());

            // Architecture should be unchanged (we didn't mutate it).
            var archAfter = roundTripped.Metadata.First(m => m.Key == "general.architecture").Value.AsString();
            Assert.Equal(originalArch, archAfter);

            // Cross-check: load through the C# wrapper around the native
            // model loader. If the file is corrupt, this will throw.
            using var model = new LlamaModel(outPath, new LlamaModelParameters
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

    [Fact]
    public void Value_Type_Mismatch_Throws()
    {
        var v = LlamaGgufValue.UInt32(7);
        Assert.Throws<InvalidOperationException>(() => v.AsString());
        Assert.Throws<InvalidOperationException>(() => v.AsUInt64());
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-gguf-file-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
