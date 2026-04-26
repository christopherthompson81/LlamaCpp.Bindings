using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tests for <see cref="LlamaLoraMerge"/>. The math-load-bearing
/// check is "scale=0 merge yields output equal to base"; if the
/// merge accumulator has any drift, this fails. Plus structural
/// checks for the validation errors.
/// </summary>
public class LoraMergeTests
{
    [Fact]
    public async Task Scale_Zero_Merge_Yields_Output_Equal_To_Base()
    {
        // Build a minimal F16 base GGUF with a single 4×6 weight tensor.
        // Build an adapter targeting that tensor with non-zero lora_a/lora_b
        // so the merge code path actually runs the matmul. With scale=0
        // the delta term goes to zero, so merged ≡ base modulo float
        // round-trip noise (we cast through F32 then back to F16).
        var dir = MakeTempDir();
        try
        {
            var basePath    = Path.Combine(dir, "base.gguf");
            var adapterPath = Path.Combine(dir, "adapter.gguf");
            var outputPath  = Path.Combine(dir, "merged.gguf");

            // Base: ne[0]=in=4, ne[1]=out=6 → flat [in*out]=24.
            // Use distinctive values so any corruption is visible.
            const int InDim = 4, OutDim = 6, Rank = 2;
            var baseValues = new float[InDim * OutDim];
            for (int i = 0; i < baseValues.Length; i++) baseValues[i] = i + 1;

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .AddTensorF32("blk.0.attn_q.weight", new long[] { InDim, OutDim }, baseValues)
                .WriteAsync(basePath);

            // Adapter: lora_a is [in=4, rank=2], lora_b is [rank=2, out=6].
            // Non-zero values so we exercise the matmul.
            var loraA = new float[InDim * Rank];
            var loraB = new float[Rank * OutDim];
            for (int i = 0; i < loraA.Length; i++) loraA[i] = (i % 3) - 1;
            for (int i = 0; i < loraB.Length; i++) loraB[i] = (i % 5) - 2;

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .SetMetadata("general.type", "adapter")
                .SetMetadata("adapter.type", "lora")
                .SetMetadata("adapter.lora.alpha", 16.0f)
                .AddTensorF32("blk.0.attn_q.weight.lora_a", new long[] { InDim, Rank }, loraA)
                .AddTensorF32("blk.0.attn_q.weight.lora_b", new long[] { Rank, OutDim }, loraB)
                .WriteAsync(adapterPath);

            // Run merge with scale=0 — merge math runs but contributes nothing.
            var result = await LlamaLoraMerge.MergeAsync(
                basePath,
                new[] { new LlamaLoraAdapterInput(adapterPath, Scale: 0f) },
                outputPath,
                new LlamaLoraMergeOptions { OutputType = LlamaHfConvertOutputType.F32 });

            Assert.Equal(1, result.TensorsTotal);
            Assert.Equal(1, result.TensorsMerged);
            Assert.Equal(0, result.TensorsCopied);

            // Read back the merged tensor and compare to base byte-for-byte.
            var merged = LlamaGgufFile.Open(outputPath);
            var t = merged.Tensors.First(x => x.Name == "blk.0.attn_q.weight");
            Assert.Equal(0u, t.TypeId);  // F32 output as requested
            Assert.Equal(InDim * OutDim * 4, t.ByteSize);

            var fileBytes = File.ReadAllBytes(outputPath);
            var dataStart = merged.DataSectionFileOffset + t.ByteOffsetInDataSection;
            var roundTripped = new float[InDim * OutDim];
            Buffer.BlockCopy(fileBytes, (int)dataStart, roundTripped, 0, roundTripped.Length * 4);
            for (int i = 0; i < baseValues.Length; i++)
            {
                Assert.Equal(baseValues[i], roundTripped[i]);
            }
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Merge_Computes_Expected_Delta_For_Hand_Computed_Case()
    {
        // 2×3 base with rank-1 adapter: easy to verify by hand.
        //   base[i, o] = (i + 1) * 10 + o            // ints 11..23
        //   lora_a[i, 0] = i + 1                     // [1, 2]
        //   lora_b[0, o] = o + 1                     // [1, 2, 3]
        //   delta[i, o] = lora_a[i, 0] * lora_b[0, o]
        //              = (i + 1) * (o + 1)
        //   alpha=4, rank=1, scale=2
        //   merge_scale = 2 * 4 / 1 = 8
        //   merged[i, o] = base[i, o] + 8 * delta[i, o]
        var dir = MakeTempDir();
        try
        {
            var basePath    = Path.Combine(dir, "base.gguf");
            var adapterPath = Path.Combine(dir, "adapter.gguf");
            var outputPath  = Path.Combine(dir, "merged.gguf");

            const int InDim = 2, OutDim = 3, Rank = 1;
            var baseValues = new float[InDim * OutDim];
            for (int o = 0; o < OutDim; o++)
                for (int i = 0; i < InDim; i++)
                    baseValues[i + o * InDim] = (i + 1) * 10 + o;

            var loraA = new float[InDim * Rank];   // [in, rank]
            for (int i = 0; i < InDim; i++) loraA[i] = i + 1;     // i + r*in, r=0

            var loraB = new float[Rank * OutDim];  // [rank, out]
            for (int o = 0; o < OutDim; o++) loraB[o * Rank] = o + 1; // r + o*rank, r=0

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .AddTensorF32("blk.0.attn_q.weight", new long[] { InDim, OutDim }, baseValues)
                .WriteAsync(basePath);

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .SetMetadata("general.type", "adapter")
                .SetMetadata("adapter.type", "lora")
                .SetMetadata("adapter.lora.alpha", 4.0f)
                .AddTensorF32("blk.0.attn_q.weight.lora_a", new long[] { InDim, Rank }, loraA)
                .AddTensorF32("blk.0.attn_q.weight.lora_b", new long[] { Rank, OutDim }, loraB)
                .WriteAsync(adapterPath);

            await LlamaLoraMerge.MergeAsync(
                basePath,
                new[] { new LlamaLoraAdapterInput(adapterPath, Scale: 2f) },
                outputPath,
                new LlamaLoraMergeOptions { OutputType = LlamaHfConvertOutputType.F32 });

            var merged = LlamaGgufFile.Open(outputPath);
            var t = merged.Tensors.First(x => x.Name == "blk.0.attn_q.weight");
            var fileBytes = File.ReadAllBytes(outputPath);
            var roundTripped = new float[InDim * OutDim];
            Buffer.BlockCopy(fileBytes, (int)(merged.DataSectionFileOffset + t.ByteOffsetInDataSection),
                roundTripped, 0, roundTripped.Length * 4);

            for (int o = 0; o < OutDim; o++)
            {
                for (int i = 0; i < InDim; i++)
                {
                    float expected = baseValues[i + o * InDim] + 8f * (i + 1) * (o + 1);
                    Assert.Equal(expected, roundTripped[i + o * InDim], 5);
                }
            }
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Untouched_Tensors_Pass_Through_Verbatim()
    {
        var dir = MakeTempDir();
        try
        {
            var basePath    = Path.Combine(dir, "base.gguf");
            var adapterPath = Path.Combine(dir, "adapter.gguf");
            var outputPath  = Path.Combine(dir, "merged.gguf");

            // Base has two tensors; adapter only targets one.
            var targeted = new float[6] { 1, 2, 3, 4, 5, 6 };
            var untouched = new float[4] { 100, 200, 300, 400 };
            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .AddTensorF32("blk.0.attn_q.weight", new long[] { 2, 3 }, targeted)
                .AddTensorF32("output_norm.weight", new long[] { 4 }, untouched)
                .WriteAsync(basePath);

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .SetMetadata("general.type", "adapter")
                .SetMetadata("adapter.type", "lora")
                .SetMetadata("adapter.lora.alpha", 1.0f)
                .AddTensorF32("blk.0.attn_q.weight.lora_a", new long[] { 2, 1 }, new float[2] { 0, 0 })
                .AddTensorF32("blk.0.attn_q.weight.lora_b", new long[] { 1, 3 }, new float[3] { 0, 0, 0 })
                .WriteAsync(adapterPath);

            var result = await LlamaLoraMerge.MergeAsync(
                basePath,
                new[] { new LlamaLoraAdapterInput(adapterPath) },
                outputPath,
                new LlamaLoraMergeOptions { OutputType = LlamaHfConvertOutputType.F32 });

            Assert.Equal(2, result.TensorsTotal);
            Assert.Equal(1, result.TensorsMerged);
            Assert.Equal(1, result.TensorsCopied);

            var merged = LlamaGgufFile.Open(outputPath);
            var u = merged.Tensors.First(x => x.Name == "output_norm.weight");
            // Untouched tensor should be byte-exact (and same type as
            // the base — F32, since we wrote it as F32 above).
            Assert.Equal(0u, u.TypeId);
            var fileBytes = File.ReadAllBytes(outputPath);
            var roundTripped = new float[4];
            Buffer.BlockCopy(fileBytes, (int)(merged.DataSectionFileOffset + u.ByteOffsetInDataSection),
                roundTripped, 0, 4 * 4);
            Assert.Equal(untouched, roundTripped);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Rejects_Non_Adapter_File()
    {
        var dir = MakeTempDir();
        try
        {
            var basePath = Path.Combine(dir, "base.gguf");
            var fakeAdapter = Path.Combine(dir, "fake.gguf");
            var outputPath = Path.Combine(dir, "out.gguf");

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .AddTensorF32("t", new long[] { 4 }, new float[4])
                .WriteAsync(basePath);

            // Adapter without general.type/adapter.type — should fail validation.
            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "test_arch")
                .AddTensorF32("t.lora_a", new long[] { 4, 1 }, new float[4])
                .AddTensorF32("t.lora_b", new long[] { 1, 1 }, new float[1])
                .WriteAsync(fakeAdapter);

            await Assert.ThrowsAnyAsync<InvalidDataException>(() =>
                LlamaLoraMerge.MergeAsync(
                    basePath,
                    new[] { new LlamaLoraAdapterInput(fakeAdapter) },
                    outputPath));
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Rejects_Architecture_Mismatch()
    {
        var dir = MakeTempDir();
        try
        {
            var basePath = Path.Combine(dir, "base.gguf");
            var adapterPath = Path.Combine(dir, "adapter.gguf");
            var outputPath = Path.Combine(dir, "out.gguf");

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "llama")
                .AddTensorF32("t", new long[] { 4 }, new float[4])
                .WriteAsync(basePath);

            await new LlamaGgufWriter()
                .SetMetadata("general.architecture", "qwen3")  // mismatch
                .SetMetadata("general.type", "adapter")
                .SetMetadata("adapter.type", "lora")
                .AddTensorF32("t.lora_a", new long[] { 4, 1 }, new float[4])
                .AddTensorF32("t.lora_b", new long[] { 1, 1 }, new float[1])
                .WriteAsync(adapterPath);

            await Assert.ThrowsAnyAsync<InvalidDataException>(() =>
                LlamaLoraMerge.MergeAsync(
                    basePath,
                    new[] { new LlamaLoraAdapterInput(adapterPath) },
                    outputPath));
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-lora-merge-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
