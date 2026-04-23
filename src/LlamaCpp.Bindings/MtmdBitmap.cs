using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;
using LlamaCpp.Bindings.Native.SafeHandles;

namespace LlamaCpp.Bindings;

/// <summary>
/// A preprocessed pixel (or, in a later phase, audio) buffer ready for
/// <c>mtmd_tokenize</c>. Construct via the <see cref="FromFile"/>,
/// <see cref="FromBytes"/>, or <see cref="FromPixels"/> factories — the ctor is
/// private because every path needs a native allocation.
/// </summary>
public sealed class MtmdBitmap : IDisposable
{
    private readonly SafeMtmdBitmapHandle _handle;
    private bool _disposed;

    internal SafeMtmdBitmapHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    /// <summary>Pixel width (for image bitmaps) or 0 (audio).</summary>
    public int Width { get; }

    /// <summary>Pixel height (for image bitmaps) or 0 (audio).</summary>
    public int Height { get; }

    /// <summary>Size of the native data buffer in bytes.</summary>
    public long ByteCount { get; }

    /// <summary>True if this bitmap carries audio PCM-F32 rather than pixels.</summary>
    public bool IsAudio { get; }

    private MtmdBitmap(SafeMtmdBitmapHandle handle)
    {
        _handle = handle;
        var raw = handle.DangerousHandle;
        Width     = (int)NativeMethods.mtmd_bitmap_get_nx(raw);
        Height    = (int)NativeMethods.mtmd_bitmap_get_ny(raw);
        ByteCount = (long)NativeMethods.mtmd_bitmap_get_n_bytes(raw);
        IsAudio   = NativeMethods.mtmd_bitmap_is_audio(raw);
    }

    /// <summary>
    /// Optional string identifier useful for KV-cache tracking — callers can
    /// hash the pixel data themselves and tag the bitmap so downstream cache
    /// layers can dedupe. Reads/writes the native <c>id</c> field directly.
    /// </summary>
    public string? Id
    {
        get
        {
            var ptr = NativeMethods.mtmd_bitmap_get_id(Handle.DangerousHandle);
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            NativeMethods.mtmd_bitmap_set_id(Handle.DangerousHandle, value);
        }
    }

    /// <summary>
    /// Load an image (or audio) file from disk using the mtmd helper, which
    /// auto-detects format via magic bytes and runs the appropriate preprocessor
    /// (stb_image for images, miniaudio for audio). The bitmap is independent of
    /// <paramref name="context"/> once returned.
    /// </summary>
    public static MtmdBitmap FromFile(MtmdContext context, string path)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Bitmap source not found: {path}", path);

        var raw = NativeMethods.mtmd_helper_bitmap_init_from_file(
            context.Handle.DangerousHandle, path);
        if (raw == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.mtmd_helper_bitmap_init_from_file),
                $"mtmd_helper_bitmap_init_from_file failed for '{path}'. " +
                $"Check the native log for the decoder error.");
        }
        return new MtmdBitmap(SafeMtmdBitmapHandle.FromUnsafeHandle(raw));
    }

    /// <summary>
    /// Decode an image/audio file already in memory (e.g. from a clipboard
    /// paste or HTTP upload). Format auto-detected via magic bytes.
    /// </summary>
    public static unsafe MtmdBitmap FromBytes(MtmdContext context, ReadOnlySpan<byte> data)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (data.IsEmpty) throw new ArgumentException("Bitmap data is empty.", nameof(data));

        fixed (byte* ptr = data)
        {
            var raw = NativeMethods.mtmd_helper_bitmap_init_from_buf(
                context.Handle.DangerousHandle, ptr, (nuint)data.Length);
            if (raw == IntPtr.Zero)
            {
                throw new LlamaException(
                    nameof(NativeMethods.mtmd_helper_bitmap_init_from_buf),
                    "mtmd_helper_bitmap_init_from_buf failed. The buffer may not be a recognised " +
                    "image/audio format, or decoding failed. Check the native log.");
            }
            return new MtmdBitmap(SafeMtmdBitmapHandle.FromUnsafeHandle(raw));
        }
    }

    /// <summary>
    /// Build a bitmap directly from an RGB pixel buffer. Bypasses the format
    /// decoders — useful when the caller has already decoded (e.g. from a
    /// <c>SKBitmap</c> or an Avalonia render target).
    /// </summary>
    /// <param name="width">Pixel width. Must be positive.</param>
    /// <param name="height">Pixel height. Must be positive.</param>
    /// <param name="rgb">Tightly packed RGB bytes — length must equal width * height * 3.</param>
    public static unsafe MtmdBitmap FromPixels(int width, int height, ReadOnlySpan<byte> rgb)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        var expected = checked((long)width * height * 3);
        if (rgb.Length != expected)
        {
            throw new ArgumentException(
                $"RGB buffer length {rgb.Length} does not match width*height*3 = {expected}.",
                nameof(rgb));
        }

        fixed (byte* ptr = rgb)
        {
            var raw = NativeMethods.mtmd_bitmap_init((uint)width, (uint)height, ptr);
            if (raw == IntPtr.Zero)
            {
                throw new LlamaException(
                    nameof(NativeMethods.mtmd_bitmap_init),
                    "mtmd_bitmap_init returned NULL.");
            }
            return new MtmdBitmap(SafeMtmdBitmapHandle.FromUnsafeHandle(raw));
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
