using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Round-trip tests for <see cref="LlamaGgufWriter"/>. We write small
/// GGUFs in pure C#, then parse them back via the native
/// <c>gguf_init_from_file</c> path to verify the layout is byte-correct.
/// If these pass, anything ggml-aware (llama-imatrix, llama-quantize,
/// the model loader) can also load the writer's output.
/// </summary>
public class GgufWriterTests
{
    [Fact]
    public async Task Write_Empty_Then_Native_Reader_Parses()
    {
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var path = Path.Combine(dir, "empty.gguf");
            await new LlamaGgufWriter()
                .SetMetadata("general.type", "test")
                .WriteAsync(path);

            using var ctx = OpenGguf(path);
            Assert.Equal(0, NativeMethods.gguf_get_n_tensors(ctx.Handle));
            // We auto-add general.alignment, plus the caller's general.type — so 2.
            Assert.Equal(2, NativeMethods.gguf_get_n_kv(ctx.Handle));

            var typeKeyId = NativeMethods.gguf_find_key(ctx.Handle, "general.type");
            Assert.True(typeKeyId >= 0);
            var typeStr = ReadString(NativeMethods.gguf_get_val_str(ctx.Handle, typeKeyId));
            Assert.Equal("test", typeStr);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Write_Tensors_Round_Trip_Through_Native_Reader()
    {
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var path = Path.Combine(dir, "tensors.gguf");

            // Two F32 tensors of different shapes. ggml convention: shape[0]
            // is the fastest-varying axis.
            var a = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };               // 3×2
            var b = new float[] { 0.5f, 1.5f, 2.5f, 3.5f };               // 4×1
            await new LlamaGgufWriter()
                .SetMetadata("test.kv", 42u)
                .AddTensorF32("alpha", new long[] { 3, 2 }, a)
                .AddTensorF32("beta",  new long[] { 4 },     b)
                .WriteAsync(path);

            using var ctx = OpenGguf(path);
            Assert.Equal(2, NativeMethods.gguf_get_n_tensors(ctx.Handle));

            var nameAlpha = ReadString(NativeMethods.gguf_get_tensor_name(ctx.Handle, 0));
            var nameBeta  = ReadString(NativeMethods.gguf_get_tensor_name(ctx.Handle, 1));
            Assert.Equal("alpha", nameAlpha);
            Assert.Equal("beta",  nameBeta);

            Assert.Equal(ggml_type.GGML_TYPE_F32, NativeMethods.gguf_get_tensor_type(ctx.Handle, 0));
            Assert.Equal(ggml_type.GGML_TYPE_F32, NativeMethods.gguf_get_tensor_type(ctx.Handle, 1));
            // alpha: 3*2*4=24 bytes; beta: 4*4=16 bytes.
            Assert.Equal((nuint)24, NativeMethods.gguf_get_tensor_size(ctx.Handle, 0));
            Assert.Equal((nuint)16, NativeMethods.gguf_get_tensor_size(ctx.Handle, 1));

            // KV scalar round-trip.
            var keyId = NativeMethods.gguf_find_key(ctx.Handle, "test.kv");
            Assert.True(keyId >= 0);
            Assert.Equal(42u, NativeMethods.gguf_get_val_u32(ctx.Handle, keyId));

            // Read tensor data from the on-disk offsets and verify bytes.
            var dataOffset = (long)NativeMethods.gguf_get_data_offset(ctx.Handle);
            var alphaOff = (long)NativeMethods.gguf_get_tensor_offset(ctx.Handle, 0);
            var betaOff  = (long)NativeMethods.gguf_get_tensor_offset(ctx.Handle, 1);

            var raw = await File.ReadAllBytesAsync(path, TestContext.Current.CancellationToken);
            AssertF32SpanEqual(a, raw, dataOffset + alphaOff);
            AssertF32SpanEqual(b, raw, dataOffset + betaOff);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Write_String_Array_Round_Trips()
    {
        LlamaBackend.Initialize();

        var dir = MakeTempDir();
        try
        {
            var path = Path.Combine(dir, "strarr.gguf");
            await new LlamaGgufWriter()
                .SetMetadataStringArray("imatrix.datasets", new[] { "wiki.test.raw", "supplemental.txt" })
                .WriteAsync(path);

            using var ctx = OpenGguf(path);
            var keyId = NativeMethods.gguf_find_key(ctx.Handle, "imatrix.datasets");
            Assert.True(keyId >= 0);
            // Type at the KV slot is GGUF_TYPE_ARRAY (=9), inner type is STRING (=8).
            Assert.Equal(9u, NativeMethods.gguf_get_kv_type(ctx.Handle, keyId));
            Assert.Equal(8u, NativeMethods.gguf_get_arr_type(ctx.Handle, keyId));
            Assert.Equal((nuint)2, NativeMethods.gguf_get_arr_n(ctx.Handle, keyId));

            var s0 = ReadString(NativeMethods.gguf_get_arr_str(ctx.Handle, keyId, 0));
            var s1 = ReadString(NativeMethods.gguf_get_arr_str(ctx.Handle, keyId, 1));
            Assert.Equal("wiki.test.raw",   s0);
            Assert.Equal("supplemental.txt", s1);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public void Constructor_Rejects_Bad_Alignment()
    {
        Assert.Throws<ArgumentException>(() => new LlamaGgufWriter(0));
        Assert.Throws<ArgumentException>(() => new LlamaGgufWriter(31));
    }

    [Fact]
    public void AddTensor_Rejects_Mismatched_Data_Length()
    {
        var w = new LlamaGgufWriter();
        Assert.Throws<ArgumentException>(() =>
            w.AddTensorF32("bad", new long[] { 3, 2 }, new float[] { 1, 2, 3 }));
    }

    // ----- helpers -----

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-gguf-writer-" + Guid.NewGuid().ToString("N"));
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
                $"gguf_init_from_file returned NULL for '{path}' — file is malformed.");
        }
        return new GgufHandle(ctx);
    }

    private static string ReadString(IntPtr ptr) =>
        ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(ptr) ?? string.Empty;

    private static void AssertF32SpanEqual(float[] expected, byte[] file, long offset)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            float actual = BitConverter.ToSingle(file, (int)(offset + i * 4));
            Assert.Equal(expected[i], actual);
        }
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
