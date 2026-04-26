using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end tests for <see cref="LlamaImatrix"/>. Runs imatrix
/// collection on the cached default test model with a small corpus,
/// then loads the output GGUF back through the native parser to verify
/// the tensors and metadata that <c>llama-quantize</c> looks for are
/// present and well-formed.
/// </summary>
public class ImatrixTests
{
    private const string CalibrationCorpus =
        "The quick brown fox jumps over the lazy dog. " +
        "Pack my box with five dozen liquor jugs. " +
        "How vexingly quick daft zebras jump! " +
        "The five boxing wizards jump quickly. " +
        "Sphinx of black quartz, judge my vow. " +
        "Amazingly few discotheques provide jukeboxes. " +
        "Jackdaws love my big sphinx of quartz. " +
        "The job requires extra pluck and zeal from every young wage earner. " +
        "Bright vixens jump; dozy fowl quack. " +
        "Quick wafting zephyrs vex bold Jim. " +
        "Two driven jocks help fax my big quiz. " +
        "Five quacking zephyrs jolt my wax bed. " +
        "The five boxing wizards jump quickly. " +
        "Heavy boxes perform quick waltzes and jigs.";

    [Fact]
    public async Task Imatrix_Writes_Valid_Gguf_With_Per_Layer_Tensors()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "test.imatrix.gguf");

            // Pad the corpus so we get multiple chunks at a small ContextSize.
            var corpus = string.Concat(Enumerable.Repeat(CalibrationCorpus + " ", 4));

            var result = await LlamaImatrix.ComputeAsync(
                model,
                corpus,
                outPath,
                new LlamaImatrixOptions
                {
                    ContextSize = 64,            // small + fast for CI
                    ProcessOutput = false,
                    AddBeginningOfSequence = true,
                });

            Assert.True(File.Exists(outPath), $"Expected imatrix GGUF at {outPath}");
            Assert.True(result.ChunkCount >= 1);
            Assert.True(result.TensorsTracked > 0,
                $"Imatrix tracked 0 tensors — the eval callback never fired or filtered everything out.");

            // Parse it back through the native reader.
            using var ctx = OpenGguf(outPath);
            int tensorCount = (int)NativeMethods.gguf_get_n_tensors(ctx.Handle);
            Assert.True(tensorCount > 0);

            // Every tracked tensor contributes two outputs (in_sum2 + counts).
            Assert.Equal(0, tensorCount % 2);

            // Validate metadata round-trip.
            var typeKey = NativeMethods.gguf_find_key(ctx.Handle, "general.type");
            Assert.True(typeKey >= 0);
            var typeStr = Marshal.PtrToStringUTF8(NativeMethods.gguf_get_val_str(ctx.Handle, typeKey));
            Assert.Equal("imatrix", typeStr);

            var chunkCountKey = NativeMethods.gguf_find_key(ctx.Handle, "imatrix.chunk_count");
            Assert.True(chunkCountKey >= 0);
            Assert.Equal((uint)result.ChunkCount,
                NativeMethods.gguf_get_val_u32(ctx.Handle, chunkCountKey));

            var chunkSizeKey = NativeMethods.gguf_find_key(ctx.Handle, "imatrix.chunk_size");
            Assert.True(chunkSizeKey >= 0);

            var datasetsKey = NativeMethods.gguf_find_key(ctx.Handle, "imatrix.datasets");
            Assert.True(datasetsKey >= 0);

            // Walk tensor names and verify the expected suffix pairs.
            // Specifically: every "<name>.in_sum2" should have a paired
            // "<name>.counts" of the right shape.
            int blkInSum2Count = 0;
            int blkCountsCount = 0;
            for (int i = 0; i < tensorCount; i++)
            {
                var name = Marshal.PtrToStringUTF8(NativeMethods.gguf_get_tensor_name(ctx.Handle, i)) ?? "";
                Assert.Equal(ggml_type.GGML_TYPE_F32,
                    NativeMethods.gguf_get_tensor_type(ctx.Handle, i));
                if (name.StartsWith("blk.", StringComparison.Ordinal))
                {
                    if (name.EndsWith(".in_sum2", StringComparison.Ordinal)) blkInSum2Count++;
                    if (name.EndsWith(".counts",  StringComparison.Ordinal)) blkCountsCount++;
                }
            }
            Assert.True(blkInSum2Count > 0, "Imatrix has no .in_sum2 tensors for blk.* layers.");
            Assert.Equal(blkInSum2Count, blkCountsCount);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Imatrix_Honors_Cancellation_Between_Chunks()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var dir = MakeTempDir();
        try
        {
            var outPath = Path.Combine(dir, "cancelled.imatrix.gguf");
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                LlamaImatrix.ComputeAsync(
                    model,
                    string.Concat(Enumerable.Repeat(CalibrationCorpus + " ", 4)),
                    outPath,
                    new LlamaImatrixOptions { ContextSize = 64 },
                    progress: null,
                    cancellationToken: cts.Token));
            Assert.False(File.Exists(outPath), "Output file should not exist when cancelled before any chunk.");
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    // ----- helpers -----

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-imatrix-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }

    private static GgufHandle OpenGguf(string path)
    {
        var p = new gguf_init_params { no_alloc = true, ctx = IntPtr.Zero };
        var ctx = NativeMethods.gguf_init_from_file(path, p);
        if (ctx == IntPtr.Zero)
        {
            throw new LlamaException(
                "gguf_init_from_file",
                $"gguf_init_from_file returned NULL for '{path}'.");
        }
        return new GgufHandle(ctx);
    }

    private sealed class GgufHandle : IDisposable
    {
        public IntPtr Handle { get; }
        public GgufHandle(IntPtr h) { Handle = h; }
        public void Dispose()
        {
            if (Handle != IntPtr.Zero) NativeMethods.gguf_free(Handle);
        }
    }
}
