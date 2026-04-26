using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCpp.Bindings;

/// <summary>
/// Pure-C# writer for the GGUF v3 file format. Covers the surface needed
/// by <see cref="LlamaImatrix"/> and (eventually) the control-vector and
/// GGUF-editor tools — scalar metadata, string-array metadata, and F32
/// tensors. No native GGUF surface is required to use this writer.
/// </summary>
/// <remarks>
/// <para>
/// Format reference: <c>llama.cpp/ggml/include/gguf.h</c>. Layout is
/// little-endian throughout. Strings are <c>uint64 length + UTF-8 bytes</c>
/// with no null terminator. The data section is aligned to
/// <see cref="Alignment"/> (default 32, matching <c>GGUF_DEFAULT_ALIGNMENT</c>);
/// each tensor's data is also aligned within that section.
/// </para>
/// <para>
/// Builder semantics: add metadata + tensors in any order, then call
/// <see cref="WriteAsync"/> or <see cref="Write"/>. The same writer can be
/// emitted multiple times if the caller wants to write the same content
/// to several files.
/// </para>
/// </remarks>
public sealed class LlamaGgufWriter
{
    /// <summary>Magic bytes "GGUF" written as the first 4 bytes of the file.</summary>
    public static readonly byte[] Magic = { (byte)'G', (byte)'G', (byte)'U', (byte)'F' };

    /// <summary>GGUF version this writer emits.</summary>
    public const uint Version = 3;

    /// <summary>Default tensor-data alignment (matches <c>GGUF_DEFAULT_ALIGNMENT</c>).</summary>
    public const uint DefaultAlignment = 32;

    /// <summary>Tensor-data alignment used by this writer instance.</summary>
    public uint Alignment { get; }

    private readonly List<MetadataEntry> _metadata = new();
    private readonly List<TensorEntry> _tensors = new();

    /// <param name="alignment">
    /// Tensor data alignment, in bytes. Must be a positive power of two.
    /// The writer also emits a <c>general.alignment</c> metadata key with
    /// this value so the file is self-describing.
    /// </param>
    public LlamaGgufWriter(uint alignment = DefaultAlignment)
    {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        {
            throw new ArgumentException(
                $"alignment must be a positive power of two; got {alignment}.",
                nameof(alignment));
        }
        Alignment = alignment;
    }

    // ----- Metadata: scalars -----

    public LlamaGgufWriter SetMetadata(string key, byte    value) => Add(key, GgufType.Uint8,   BitConverter.GetBytes((ushort)value).AsSpan(0, 1).ToArray());
    public LlamaGgufWriter SetMetadata(string key, sbyte   value) => Add(key, GgufType.Int8,    new[] { unchecked((byte)value) });
    public LlamaGgufWriter SetMetadata(string key, ushort  value) => Add(key, GgufType.Uint16,  ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, short   value) => Add(key, GgufType.Int16,   ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, uint    value) => Add(key, GgufType.Uint32,  ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, int     value) => Add(key, GgufType.Int32,   ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, ulong   value) => Add(key, GgufType.Uint64,  ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, long    value) => Add(key, GgufType.Int64,   ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, float   value) => Add(key, GgufType.Float32, ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, double  value) => Add(key, GgufType.Float64, ToBytes(value));
    public LlamaGgufWriter SetMetadata(string key, bool    value) => Add(key, GgufType.Bool,    new[] { (byte)(value ? 1 : 0) });

    public LlamaGgufWriter SetMetadata(string key, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        // gguf_string: uint64 length + UTF-8 bytes.
        var utf8 = Encoding.UTF8.GetBytes(value);
        var buf = new byte[8 + utf8.Length];
        BinaryPrimitives.WriteUInt64LittleEndian(buf, (ulong)utf8.Length);
        utf8.CopyTo(buf, 8);
        return Add(key, GgufType.String, buf);
    }

    /// <summary>
    /// Set a metadata KV from a <see cref="LlamaGgufValue"/> of any type.
    /// Used by <see cref="LlamaGgufFile"/>'s save path to round-trip
    /// arbitrary values without unpacking and re-packing them.
    /// </summary>
    public LlamaGgufWriter SetMetadata(string key, LlamaGgufValue value)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNull(value);
        switch (value.Type)
        {
            case LlamaGgufType.Uint8:   return SetMetadata(key, value.AsUInt8());
            case LlamaGgufType.Int8:    return SetMetadata(key, value.AsInt8());
            case LlamaGgufType.Uint16:  return SetMetadata(key, value.AsUInt16());
            case LlamaGgufType.Int16:   return SetMetadata(key, value.AsInt16());
            case LlamaGgufType.Uint32:  return SetMetadata(key, value.AsUInt32());
            case LlamaGgufType.Int32:   return SetMetadata(key, value.AsInt32());
            case LlamaGgufType.Float32: return SetMetadata(key, value.AsFloat32());
            case LlamaGgufType.Bool:    return SetMetadata(key, value.AsBool());
            case LlamaGgufType.String:  return SetMetadata(key, value.AsString());
            case LlamaGgufType.Uint64:  return SetMetadata(key, value.AsUInt64());
            case LlamaGgufType.Int64:   return SetMetadata(key, value.AsInt64());
            case LlamaGgufType.Float64: return SetMetadata(key, value.AsFloat64());
            case LlamaGgufType.Array:
                if (value.InnerType == LlamaGgufType.String)
                {
                    return SetMetadataStringArray(key, value.AsStringArray());
                }
                return SetMetadataPrimitiveArray(key, value);
            default:
                throw new ArgumentException($"Unsupported gguf type {value.Type}", nameof(value));
        }
    }

    /// <summary>
    /// Encode a primitive-array value (any <see cref="LlamaGgufValue.InnerType"/>
    /// except <see cref="LlamaGgufType.String"/> or <see cref="LlamaGgufType.Array"/>).
    /// </summary>
    private LlamaGgufWriter SetMetadataPrimitiveArray(string key, LlamaGgufValue value)
    {
        using var ms = new MemoryStream();
        WriteUint32(ms, (uint)value.InnerType);
        long count = value.ArrayCount;
        WriteUint64(ms, (ulong)count);
        var raw = value.RawValue;
        // Each primitive type is fixed-size — emit the underlying bytes
        // verbatim. We do this rather than ReadOnlySpan<byte> via
        // MemoryMarshal so the writer doesn't need an unsafe block for
        // non-trivial alignment cases.
        switch (value.InnerType)
        {
            case LlamaGgufType.Uint8:   ms.Write((byte[])raw);   break;
            case LlamaGgufType.Int8:    foreach (var v in (sbyte[])raw)  ms.WriteByte(unchecked((byte)v)); break;
            case LlamaGgufType.Uint16:  foreach (var v in (ushort[])raw) WriteUint16(ms, v); break;
            case LlamaGgufType.Int16:   foreach (var v in (short[])raw)  WriteInt16(ms, v); break;
            case LlamaGgufType.Uint32:  foreach (var v in (uint[])raw)   WriteUint32(ms, v); break;
            case LlamaGgufType.Int32:   foreach (var v in (int[])raw)    WriteInt32(ms, v); break;
            case LlamaGgufType.Float32: foreach (var v in (float[])raw)  WriteFloat32(ms, v); break;
            case LlamaGgufType.Bool:    foreach (var v in (bool[])raw)   ms.WriteByte((byte)(v ? 1 : 0)); break;
            case LlamaGgufType.Uint64:  foreach (var v in (ulong[])raw)  WriteUint64(ms, v); break;
            case LlamaGgufType.Int64:   foreach (var v in (long[])raw)   WriteInt64(ms, v); break;
            case LlamaGgufType.Float64: foreach (var v in (double[])raw) WriteFloat64(ms, v); break;
            default:
                throw new ArgumentException($"Unsupported array inner type {value.InnerType}", nameof(value));
        }
        return Add(key, GgufType.Array, ms.ToArray());
    }

    /// <summary>
    /// Add a string-array metadata value (e.g. <c>imatrix.datasets</c>).
    /// Encoded as <c>{ inner_type=GGUF_TYPE_STRING, count=N, gguf_string × N }</c>.
    /// </summary>
    public LlamaGgufWriter SetMetadataStringArray(string key, IReadOnlyList<string> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        using var ms = new MemoryStream();
        WriteUint32(ms, (uint)GgufType.String);
        WriteUint64(ms, (ulong)values.Count);
        foreach (var v in values)
        {
            ArgumentNullException.ThrowIfNull(v);
            var utf8 = Encoding.UTF8.GetBytes(v);
            WriteUint64(ms, (ulong)utf8.Length);
            ms.Write(utf8);
        }
        return Add(key, GgufType.Array, ms.ToArray());
    }

    // ----- Tensors -----

    /// <summary>
    /// Add an F32 tensor to the file. <paramref name="shape"/> is in GGML
    /// convention: <c>shape[0]</c> is the fastest-varying ("row width"),
    /// <c>shape[1]</c> the next, etc. <paramref name="data"/> length must
    /// equal <c>product(shape)</c>.
    /// </summary>
    public LlamaGgufWriter AddTensorF32(string name, ReadOnlySpan<long> shape, ReadOnlySpan<float> data)
    {
        ValidateTensorBasics(name, shape);
        long expected = ProductOf(shape);
        if (data.Length != expected)
        {
            throw new ArgumentException(
                $"data length {data.Length} != product(shape) {expected}.", nameof(data));
        }

        var bytes = new byte[data.Length * sizeof(float)];
        MemoryMarshal.AsBytes(data).CopyTo(bytes);
        _tensors.Add(new TensorEntry(name, shape.ToArray(), TypeId: 0, new InMemoryTensorSource(bytes)));
        return this;
    }

    /// <summary>
    /// Add a tensor of any <c>ggml_type</c> from in-memory bytes. The
    /// <paramref name="typeId"/> is the wire-format integer (e.g. 0 = F32,
    /// 1 = F16, 12 = Q4_K). Used by the GGUF Editor to round-trip tensors
    /// whose type isn't exposed via <see cref="AddTensorF32"/>.
    /// </summary>
    public LlamaGgufWriter AddTensor(string name, uint typeId, ReadOnlySpan<long> shape, byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        ValidateTensorBasics(name, shape);
        _tensors.Add(new TensorEntry(name, shape.ToArray(), typeId, new InMemoryTensorSource(data)));
        return this;
    }

    /// <summary>
    /// Add a tensor whose data lives in another file. At write time the
    /// writer streams the bytes from <paramref name="sourcePath"/>
    /// starting at <paramref name="sourceOffsetInFile"/>. Used by
    /// <see cref="LlamaGgufFile.SaveAsAsync"/> so a GGUF edit doesn't
    /// need to load multi-GB tensor blocks into managed memory.
    /// </summary>
    public LlamaGgufWriter AddTensorFromFile(
        string name, uint typeId, ReadOnlySpan<long> shape,
        string sourcePath, long sourceOffsetInFile, long byteSize)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ValidateTensorBasics(name, shape);
        if (sourceOffsetInFile < 0)
            throw new ArgumentOutOfRangeException(nameof(sourceOffsetInFile), "Must be >= 0.");
        if (byteSize < 0)
            throw new ArgumentOutOfRangeException(nameof(byteSize), "Must be >= 0.");
        _tensors.Add(new TensorEntry(name, shape.ToArray(), typeId,
            new FileBackedTensorSource(sourcePath, sourceOffsetInFile, byteSize)));
        return this;
    }

    private static void ValidateTensorBasics(string name, ReadOnlySpan<long> shape)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (shape.Length is < 1 or > 4)
        {
            throw new ArgumentException(
                $"shape must have 1..4 dimensions; got {shape.Length}.", nameof(shape));
        }
        foreach (var d in shape)
        {
            if (d <= 0)
            {
                throw new ArgumentException(
                    $"All dims must be positive; got {d}.", nameof(shape));
            }
        }
    }

    private static long ProductOf(ReadOnlySpan<long> shape)
    {
        long p = 1;
        foreach (var d in shape) p *= d;
        return p;
    }

    // ----- Emit -----

    public async Task WriteAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var tmp = path + ".tmp";
        try
        {
            await using (var fs = File.Create(tmp))
            {
                await WriteAsyncCore(fs, cancellationToken);
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            if (File.Exists(tmp))
            {
                try { File.Delete(tmp); } catch { /* best-effort */ }
            }
            throw;
        }
    }

    public void Write(Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        WriteAsyncCore(destination, CancellationToken.None).GetAwaiter().GetResult();
    }

    private async Task WriteAsyncCore(Stream dest, CancellationToken cancellationToken)
    {
        // Auto-add general.alignment so the file is self-describing.
        // We avoid duplicates if the caller already set one.
        bool hasAlignment = false;
        foreach (var m in _metadata)
        {
            if (m.Key == "general.alignment") { hasAlignment = true; break; }
        }
        var metadata = hasAlignment
            ? _metadata
            : _metadata.Concat(new[] { Encode("general.alignment", GgufType.Uint32, ToBytes(Alignment)) }).ToList();

        // Header.
        await dest.WriteAsync(Magic, cancellationToken);
        await WriteUint32Async(dest, Version, cancellationToken);
        await WriteUint64Async(dest, (ulong)_tensors.Count, cancellationToken);
        await WriteUint64Async(dest, (ulong)metadata.Count, cancellationToken);

        // Metadata KV section.
        foreach (var m in metadata)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await WriteGgufStringAsync(dest, m.Key, cancellationToken);
            await WriteUint32Async(dest, (uint)m.Type, cancellationToken);
            await dest.WriteAsync(m.Value, cancellationToken);
        }

        // Tensor info section.
        // Each tensor's data offset (relative to the data section start) is
        // the running aligned cursor over the prior tensors' byte sizes.
        long cursor = 0;
        var offsets = new long[_tensors.Count];
        for (int i = 0; i < _tensors.Count; i++)
        {
            offsets[i] = cursor;
            cursor = AlignUp(cursor + _tensors[i].Data.ByteSize, Alignment);
        }
        for (int i = 0; i < _tensors.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var t = _tensors[i];
            await WriteGgufStringAsync(dest, t.Name, cancellationToken);
            await WriteUint32Async(dest, (uint)t.Shape.Length, cancellationToken);
            foreach (var d in t.Shape)
            {
                await WriteUint64Async(dest, (ulong)d, cancellationToken);
            }
            await WriteUint32Async(dest, t.TypeId, cancellationToken);
            await WriteUint64Async(dest, (ulong)offsets[i], cancellationToken);
        }

        // Pad up to the data-section start (must be aligned to Alignment).
        var headerEnd = dest.CanSeek ? dest.Position : -1;
        if (headerEnd >= 0)
        {
            var padded = AlignUp(headerEnd, Alignment);
            await WriteZerosAsync(dest, (int)(padded - headerEnd), cancellationToken);
        }
        else
        {
            // Non-seekable streams aren't a real use-case for our writer,
            // but be explicit rather than producing a broken file.
            throw new NotSupportedException(
                "GGUF writer requires a seekable destination stream.");
        }

        // Tensor data section. Each tensor copies its bytes (in-memory or
        // streamed from a source file), then we pad to the next tensor's
        // declared offset so offsets[i+1] is honored byte-for-byte.
        for (int i = 0; i < _tensors.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var src = _tensors[i].Data;
            await src.CopyToAsync(dest, cancellationToken);
            long nextEnd = (i + 1 < _tensors.Count)
                ? offsets[i + 1]
                : AlignUp(src.ByteSize, Alignment);
            long pad = nextEnd - (offsets[i] + src.ByteSize);
            await WriteZerosAsync(dest, (int)pad, cancellationToken);
        }
    }

    // ----- Helpers -----

    private LlamaGgufWriter Add(string key, GgufType type, byte[] valueBytes)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        _metadata.Add(Encode(key, type, valueBytes));
        return this;
    }

    private static MetadataEntry Encode(string key, GgufType type, byte[] valueBytes) =>
        new(key, type, valueBytes);

    private static byte[] ToBytes(ushort v) { var b = new byte[2]; BinaryPrimitives.WriteUInt16LittleEndian(b, v); return b; }
    private static byte[] ToBytes(short  v) { var b = new byte[2]; BinaryPrimitives.WriteInt16LittleEndian(b, v);  return b; }
    private static byte[] ToBytes(uint   v) { var b = new byte[4]; BinaryPrimitives.WriteUInt32LittleEndian(b, v); return b; }
    private static byte[] ToBytes(int    v) { var b = new byte[4]; BinaryPrimitives.WriteInt32LittleEndian(b, v);  return b; }
    private static byte[] ToBytes(ulong  v) { var b = new byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(b, v); return b; }
    private static byte[] ToBytes(long   v) { var b = new byte[8]; BinaryPrimitives.WriteInt64LittleEndian(b, v);  return b; }
    private static byte[] ToBytes(float  v) { var b = new byte[4]; BinaryPrimitives.WriteSingleLittleEndian(b, v); return b; }
    private static byte[] ToBytes(double v) { var b = new byte[8]; BinaryPrimitives.WriteDoubleLittleEndian(b, v); return b; }

    private static void WriteUint16( Stream s, ushort v) { Span<byte> b = stackalloc byte[2]; BinaryPrimitives.WriteUInt16LittleEndian(b, v); s.Write(b); }
    private static void WriteInt16(  Stream s, short v)  { Span<byte> b = stackalloc byte[2]; BinaryPrimitives.WriteInt16LittleEndian(b, v);  s.Write(b); }
    private static void WriteUint32( Stream s, uint v)   { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteUInt32LittleEndian(b, v); s.Write(b); }
    private static void WriteInt32(  Stream s, int v)    { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteInt32LittleEndian(b, v);  s.Write(b); }
    private static void WriteFloat32(Stream s, float v)  { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteSingleLittleEndian(b, v); s.Write(b); }
    private static void WriteUint64( Stream s, ulong v)  { Span<byte> b = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(b, v); s.Write(b); }
    private static void WriteInt64(  Stream s, long v)   { Span<byte> b = stackalloc byte[8]; BinaryPrimitives.WriteInt64LittleEndian(b, v);  s.Write(b); }
    private static void WriteFloat64(Stream s, double v) { Span<byte> b = stackalloc byte[8]; BinaryPrimitives.WriteDoubleLittleEndian(b, v); s.Write(b); }

    private static async Task WriteUint32Async(Stream s, uint v, CancellationToken ct)
    {
        var b = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(b, v);
        await s.WriteAsync(b, ct);
    }
    private static async Task WriteUint64Async(Stream s, ulong v, CancellationToken ct)
    {
        var b = new byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(b, v);
        await s.WriteAsync(b, ct);
    }
    private static async Task WriteGgufStringAsync(Stream s, string str, CancellationToken ct)
    {
        var utf8 = Encoding.UTF8.GetBytes(str);
        await WriteUint64Async(s, (ulong)utf8.Length, ct);
        await s.WriteAsync(utf8, ct);
    }
    private static async Task WriteZerosAsync(Stream s, int count, CancellationToken ct)
    {
        if (count <= 0) return;
        var b = new byte[count];
        await s.WriteAsync(b, ct);
    }

    private static long AlignUp(long value, uint alignment) =>
        ((value + alignment - 1) / alignment) * alignment;

    // ----- Internal types -----

    private enum GgufType : uint
    {
        Uint8   = 0,
        Int8    = 1,
        Uint16  = 2,
        Int16   = 3,
        Uint32  = 4,
        Int32   = 5,
        Float32 = 6,
        Bool    = 7,
        String  = 8,
        Array   = 9,
        Uint64  = 10,
        Int64   = 11,
        Float64 = 12,
    }

    private sealed record MetadataEntry(string Key, GgufType Type, byte[] Value);

    /// <summary>
    /// One tensor's wire-format metadata plus a polymorphic data source
    /// (in-memory <see cref="InMemoryTensorSource"/> or
    /// <see cref="FileBackedTensorSource"/>). The data source is opened
    /// at write time so 4 GB weight blocks don't sit in managed memory
    /// while the writer is being built.
    /// </summary>
    private sealed record TensorEntry(string Name, long[] Shape, uint TypeId, ITensorSource Data);

    private interface ITensorSource
    {
        long ByteSize { get; }
        Task CopyToAsync(Stream dest, CancellationToken ct);
    }

    private sealed class InMemoryTensorSource : ITensorSource
    {
        private readonly byte[] _bytes;
        public InMemoryTensorSource(byte[] bytes) { _bytes = bytes; }
        public long ByteSize => _bytes.LongLength;
        public Task CopyToAsync(Stream dest, CancellationToken ct) => dest.WriteAsync(_bytes, ct).AsTask();
    }

    private sealed class FileBackedTensorSource : ITensorSource
    {
        private readonly string _path;
        private readonly long _offset;
        public long ByteSize { get; }
        public FileBackedTensorSource(string path, long offset, long byteSize)
        {
            _path = path;
            _offset = offset;
            ByteSize = byteSize;
        }
        public async Task CopyToAsync(Stream dest, CancellationToken ct)
        {
            // Open the source per call so concurrent writers aren't a
            // problem and the FileStream gets disposed predictably.
            await using var src = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.Read);
            src.Seek(_offset, SeekOrigin.Begin);

            var buf = new byte[Math.Min(ByteSize, 1 << 20)];
            long remaining = ByteSize;
            while (remaining > 0)
            {
                ct.ThrowIfCancellationRequested();
                int want = (int)Math.Min(buf.Length, remaining);
                int n = await src.ReadAsync(buf.AsMemory(0, want), ct);
                if (n <= 0)
                {
                    throw new EndOfStreamException(
                        $"Source GGUF '{_path}' ended early while streaming tensor at offset {_offset}: " +
                        $"{ByteSize - remaining}/{ByteSize} bytes copied.");
                }
                await dest.WriteAsync(buf.AsMemory(0, n), ct);
                remaining -= n;
            }
        }
    }
}
