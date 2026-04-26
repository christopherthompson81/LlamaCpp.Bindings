using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCpp.Bindings;

/// <summary>
/// GGUF metadata value type — mirrors <c>enum gguf_type</c> from
/// <c>gguf.h</c>. Numeric values match the wire format so they can be
/// cast directly into <see cref="LlamaGgufValue"/> serialization.
/// </summary>
public enum LlamaGgufType : uint
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

/// <summary>
/// One GGUF metadata value. Carries the wire type plus a typed payload.
/// Use the typed factories to construct (e.g. <see cref="UInt32"/>) and
/// the typed accessors to read (e.g. <see cref="AsUInt32"/>); a type
/// mismatch throws <see cref="InvalidOperationException"/>.
/// </summary>
public sealed class LlamaGgufValue
{
    /// <summary>Top-level type tag.</summary>
    public LlamaGgufType Type { get; }

    /// <summary>
    /// Inner element type when <see cref="Type"/> is <see cref="LlamaGgufType.Array"/>.
    /// Default <c>0</c> (Uint8) for non-array values; do not read.
    /// </summary>
    public LlamaGgufType InnerType { get; }

    private readonly object _value;

    private LlamaGgufValue(LlamaGgufType type, object value, LlamaGgufType innerType = LlamaGgufType.Uint8)
    {
        Type = type;
        InnerType = innerType;
        _value = value;
    }

    // ----- Scalar factories -----
    public static LlamaGgufValue UInt8  (byte    v) => new(LlamaGgufType.Uint8,   v);
    public static LlamaGgufValue Int8   (sbyte   v) => new(LlamaGgufType.Int8,    v);
    public static LlamaGgufValue UInt16 (ushort  v) => new(LlamaGgufType.Uint16,  v);
    public static LlamaGgufValue Int16  (short   v) => new(LlamaGgufType.Int16,   v);
    public static LlamaGgufValue UInt32 (uint    v) => new(LlamaGgufType.Uint32,  v);
    public static LlamaGgufValue Int32  (int     v) => new(LlamaGgufType.Int32,   v);
    public static LlamaGgufValue Float32(float   v) => new(LlamaGgufType.Float32, v);
    public static LlamaGgufValue Bool   (bool    v) => new(LlamaGgufType.Bool,    v);
    public static LlamaGgufValue String (string  v) => new(LlamaGgufType.String,  v ?? throw new ArgumentNullException(nameof(v)));
    public static LlamaGgufValue UInt64 (ulong   v) => new(LlamaGgufType.Uint64,  v);
    public static LlamaGgufValue Int64  (long    v) => new(LlamaGgufType.Int64,   v);
    public static LlamaGgufValue Float64(double  v) => new(LlamaGgufType.Float64, v);

    // ----- Array factories -----

    /// <summary>Build a string-array value (typed inner = String).</summary>
    public static LlamaGgufValue StringArray(IReadOnlyList<string> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var copy = new string[values.Count];
        for (int i = 0; i < copy.Length; i++)
            copy[i] = values[i] ?? throw new ArgumentException($"values[{i}] is null", nameof(values));
        return new LlamaGgufValue(LlamaGgufType.Array, copy, LlamaGgufType.String);
    }

    /// <summary>
    /// Build a primitive-array value. <typeparamref name="T"/> must be one of
    /// <c>byte/sbyte/ushort/short/uint/int/ulong/long/float/double/bool</c>.
    /// </summary>
    public static LlamaGgufValue PrimitiveArray<T>(IReadOnlyList<T> values) where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(values);
        var inner = InferPrimitiveInner<T>();
        var arr = new T[values.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = values[i];
        return new LlamaGgufValue(LlamaGgufType.Array, arr, inner);
    }

    // ----- Scalar accessors -----
    public byte    AsUInt8()   { Require(LlamaGgufType.Uint8);   return (byte)_value; }
    public sbyte   AsInt8()    { Require(LlamaGgufType.Int8);    return (sbyte)_value; }
    public ushort  AsUInt16()  { Require(LlamaGgufType.Uint16);  return (ushort)_value; }
    public short   AsInt16()   { Require(LlamaGgufType.Int16);   return (short)_value; }
    public uint    AsUInt32()  { Require(LlamaGgufType.Uint32);  return (uint)_value; }
    public int     AsInt32()   { Require(LlamaGgufType.Int32);   return (int)_value; }
    public float   AsFloat32() { Require(LlamaGgufType.Float32); return (float)_value; }
    public bool    AsBool()    { Require(LlamaGgufType.Bool);    return (bool)_value; }
    public string  AsString()  { Require(LlamaGgufType.String);  return (string)_value; }
    public ulong   AsUInt64()  { Require(LlamaGgufType.Uint64);  return (ulong)_value; }
    public long    AsInt64()   { Require(LlamaGgufType.Int64);   return (long)_value; }
    public double  AsFloat64() { Require(LlamaGgufType.Float64); return (double)_value; }

    // ----- Array accessors -----

    public IReadOnlyList<string> AsStringArray()
    {
        if (Type != LlamaGgufType.Array || InnerType != LlamaGgufType.String)
            throw new InvalidOperationException($"Value is {Describe()}, not an array of strings.");
        return (string[])_value;
    }

    /// <summary>Length of the array (any inner type).</summary>
    public long ArrayCount =>
        _value is string[] s ? s.LongLength
        : _value is Array a ? a.LongLength
        : throw new InvalidOperationException($"Value is {Describe()}, not an array.");

    /// <summary>Read a primitive array as a typed array. Throws if inner type doesn't match <typeparamref name="T"/>.</summary>
    public T[] AsArray<T>() where T : unmanaged
    {
        if (Type != LlamaGgufType.Array)
            throw new InvalidOperationException($"Value is {Describe()}, not an array.");
        if (_value is not T[] arr)
            throw new InvalidOperationException(
                $"Value's inner type is {InnerType}, not {typeof(T).Name}.");
        var copy = new T[arr.Length];
        Array.Copy(arr, copy, arr.Length);
        return copy;
    }

    // ----- Display -----

    /// <summary>
    /// Human-friendly rendering for editor UIs. Strings are quoted, large
    /// arrays are summarized with their count + first few elements.
    /// </summary>
    public string ToDisplayString(int maxArrayPreview = 6)
    {
        return Type switch
        {
            LlamaGgufType.Bool    => ((bool)_value) ? "true" : "false",
            LlamaGgufType.String  => $"\"{((string)_value)}\"",
            LlamaGgufType.Array   => RenderArray(maxArrayPreview),
            _ => System.Convert.ToString(_value, System.Globalization.CultureInfo.InvariantCulture) ?? "",
        };
    }

    public override string ToString() => ToDisplayString();

    /// <summary>Internal: get the raw object (for the writer's serializer).</summary>
    internal object RawValue => _value;

    private string Describe() => Type == LlamaGgufType.Array ? $"Array<{InnerType}>" : Type.ToString();

    private void Require(LlamaGgufType expected)
    {
        if (Type != expected) throw new InvalidOperationException($"Value is {Describe()}, expected {expected}.");
    }

    private string RenderArray(int maxPreview)
    {
        long count = ArrayCount;
        var sb = new StringBuilder($"[{InnerType} × {count}]");
        if (count == 0) return sb.ToString();

        sb.Append(" { ");
        int shown = (int)Math.Min(count, maxPreview);
        for (int i = 0; i < shown; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(InnerType == LlamaGgufType.String
                ? "\"" + ((string[])_value)[i] + "\""
                : ((Array)_value).GetValue(i)?.ToString());
        }
        if (count > shown) sb.Append($", … +{count - shown}");
        sb.Append(" }");
        return sb.ToString();
    }

    private static LlamaGgufType InferPrimitiveInner<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(byte))   return LlamaGgufType.Uint8;
        if (typeof(T) == typeof(sbyte))  return LlamaGgufType.Int8;
        if (typeof(T) == typeof(ushort)) return LlamaGgufType.Uint16;
        if (typeof(T) == typeof(short))  return LlamaGgufType.Int16;
        if (typeof(T) == typeof(uint))   return LlamaGgufType.Uint32;
        if (typeof(T) == typeof(int))    return LlamaGgufType.Int32;
        if (typeof(T) == typeof(float))  return LlamaGgufType.Float32;
        if (typeof(T) == typeof(ulong))  return LlamaGgufType.Uint64;
        if (typeof(T) == typeof(long))   return LlamaGgufType.Int64;
        if (typeof(T) == typeof(double)) return LlamaGgufType.Float64;
        if (typeof(T) == typeof(bool))   return LlamaGgufType.Bool;
        throw new ArgumentException($"Type {typeof(T)} is not a supported GGUF primitive.");
    }
}

/// <summary>
/// One metadata entry on a <see cref="LlamaGgufFile"/>. Mutable —
/// editors mutate <see cref="Key"/> and <see cref="Value"/> directly.
/// </summary>
public sealed class LlamaGgufMetadataEntry
{
    public string Key { get; set; }
    public LlamaGgufValue Value { get; set; }

    public LlamaGgufMetadataEntry(string key, LlamaGgufValue value)
    {
        Key = key;
        Value = value;
    }
}

/// <summary>
/// Tensor info read from a GGUF. Immutable — editors that want to
/// rename or retype tensors should instead mutate
/// <see cref="LlamaGgufFile.Tensors"/> in place by replacing entries.
/// </summary>
public sealed class LlamaGgufTensorInfo
{
    /// <summary>Raw <c>ggml_type</c> integer. Matches the wire format.</summary>
    public uint TypeId { get; }

    /// <summary>
    /// Typed mirror, or <c>null</c> if the tensor's <see cref="TypeId"/>
    /// isn't a value defined in <see cref="LlamaTensorType"/>.
    /// </summary>
    public LlamaTensorType? Type =>
        Enum.IsDefined(typeof(LlamaTensorType), (int)TypeId) ? (LlamaTensorType)(int)TypeId : null;

    public string Name { get; set; }

    /// <summary>Element counts per dimension; <c>Dimensions[0]</c> is the fastest-varying axis (ggml convention).</summary>
    public long[] Dimensions { get; }

    /// <summary>Byte length of the tensor data on disk.</summary>
    public long ByteSize { get; }

    /// <summary>Byte offset within the source file's data section. Used by editors to stream-copy on save.</summary>
    public long ByteOffsetInDataSection { get; }

    public LlamaGgufTensorInfo(string name, uint typeId, long[] dimensions, long byteSize, long byteOffsetInDataSection)
    {
        Name = name;
        TypeId = typeId;
        Dimensions = dimensions;
        ByteSize = byteSize;
        ByteOffsetInDataSection = byteOffsetInDataSection;
    }

    /// <summary>Display form: <c>"tok_embd.weight  Q4_K  [4096,32000]"</c>.</summary>
    public override string ToString()
    {
        var typeName = Type?.ToString() ?? $"type#{TypeId}";
        return $"{Name}  {typeName}  [{string.Join(",", Dimensions)}]";
    }
}

/// <summary>
/// Pure-C# reader for the GGUF v3 file format. Materializes every
/// metadata KV and tensor info into managed objects on
/// <see cref="Open"/>. Tensor data is NOT loaded — each
/// <see cref="LlamaGgufTensorInfo"/> carries its byte offset into the
/// data section, and the editor's save path streams it from
/// <see cref="SourcePath"/> at write time.
/// </summary>
/// <remarks>
/// Format reference: <c>llama.cpp/ggml/include/gguf.h</c>. This reader
/// is symmetric with <see cref="LlamaGgufWriter"/> — round-tripping
/// through both should preserve content (modulo metadata KV ordering
/// and the auto-inserted <c>general.alignment</c>).
/// </remarks>
public sealed class LlamaGgufFile
{
    /// <summary>GGUF magic bytes "GGUF".</summary>
    public static readonly byte[] Magic = LlamaGgufWriter.Magic;

    public string SourcePath { get; }
    public uint Version { get; }
    public uint Alignment { get; }

    /// <summary>
    /// Mutable metadata list. Editors add/remove/edit entries here, then
    /// call <see cref="SaveAs"/>. Order is preserved on save.
    /// </summary>
    public List<LlamaGgufMetadataEntry> Metadata { get; }

    /// <summary>
    /// Mutable tensor info list. Currently only <see cref="LlamaGgufTensorInfo.Name"/>
    /// edits are honored at save time (rename support); type and shape
    /// changes would require re-quantizing the data, which the GGUF Editor
    /// doesn't do — use the Quantize tool for that.
    /// </summary>
    public List<LlamaGgufTensorInfo> Tensors { get; }

    /// <summary>Byte offset in the source file where the tensor data section begins.</summary>
    public long DataSectionFileOffset { get; }

    private LlamaGgufFile(string source, uint version, uint alignment,
        List<LlamaGgufMetadataEntry> metadata, List<LlamaGgufTensorInfo> tensors,
        long dataSectionFileOffset)
    {
        SourcePath = source;
        Version = version;
        Alignment = alignment;
        Metadata = metadata;
        Tensors = tensors;
        DataSectionFileOffset = dataSectionFileOffset;
    }

    /// <summary>Open a GGUF file and read its header + metadata + tensor info.</summary>
    public static LlamaGgufFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var fs = File.OpenRead(path);
        return ReadFromStream(path, fs);
    }

    private static LlamaGgufFile ReadFromStream(string path, Stream src)
    {
        Span<byte> magic = stackalloc byte[4];
        ReadExactly(src, magic);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
        {
            throw new InvalidDataException($"Not a GGUF file: magic={magic[0]:X2}{magic[1]:X2}{magic[2]:X2}{magic[3]:X2}.");
        }
        uint version = ReadUInt32(src);
        ulong tensorCount = ReadUInt64(src);
        ulong kvCount = ReadUInt64(src);

        var metadata = new List<LlamaGgufMetadataEntry>((int)Math.Min(kvCount, 4096));
        uint alignment = LlamaGgufWriter.DefaultAlignment;
        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadGgufString(src);
            var typeRaw = ReadUInt32(src);
            var type = (LlamaGgufType)typeRaw;
            var value = ReadValue(src, type);
            metadata.Add(new LlamaGgufMetadataEntry(key, value));
            // Honour an explicit general.alignment override.
            if (key == "general.alignment" && value.Type == LlamaGgufType.Uint32)
            {
                alignment = value.AsUInt32();
            }
        }

        var tensors = new List<LlamaGgufTensorInfo>((int)Math.Min(tensorCount, 1 << 20));
        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadGgufString(src);
            uint nDims = ReadUInt32(src);
            if (nDims is 0 or > 8)
            {
                throw new InvalidDataException($"Tensor '{name}' has implausible n_dims={nDims}.");
            }
            var dims = new long[nDims];
            for (int d = 0; d < nDims; d++)
            {
                dims[d] = (long)ReadUInt64(src);
            }
            uint typeId = ReadUInt32(src);
            ulong offset = ReadUInt64(src);
            long byteSize = ComputeByteSize(typeId, dims);
            tensors.Add(new LlamaGgufTensorInfo(name, typeId, dims, byteSize, (long)offset));
        }

        // Data section starts at the next aligned position.
        long dataSectionStart = AlignUp(src.Position, alignment);

        return new LlamaGgufFile(path, version, alignment, metadata, tensors, dataSectionStart);
    }

    /// <summary>
    /// Write the (potentially mutated) metadata and tensors to <paramref name="outputPath"/>,
    /// streaming tensor data from the source file. The source file is
    /// re-opened in shared-read mode so this can be called even if other
    /// processes have the file open.
    /// </summary>
    public Task SaveAsAsync(string outputPath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        var writer = new LlamaGgufWriter(Alignment);
        foreach (var kv in Metadata)
        {
            writer.SetMetadata(kv.Key, kv.Value);
        }
        foreach (var t in Tensors)
        {
            writer.AddTensorFromFile(
                name: t.Name,
                typeId: t.TypeId,
                shape: t.Dimensions,
                sourcePath: SourcePath,
                sourceOffsetInFile: DataSectionFileOffset + t.ByteOffsetInDataSection,
                byteSize: t.ByteSize);
        }
        return writer.WriteAsync(outputPath, cancellationToken);
    }

    // ----- format helpers (kept here so the file is self-contained) -----

    private static LlamaGgufValue ReadValue(Stream s, LlamaGgufType type) => type switch
    {
        LlamaGgufType.Uint8   => LlamaGgufValue.UInt8(  ReadUInt8(s)),
        LlamaGgufType.Int8    => LlamaGgufValue.Int8(   ReadInt8(s)),
        LlamaGgufType.Uint16  => LlamaGgufValue.UInt16( ReadUInt16(s)),
        LlamaGgufType.Int16   => LlamaGgufValue.Int16(  ReadInt16(s)),
        LlamaGgufType.Uint32  => LlamaGgufValue.UInt32( ReadUInt32(s)),
        LlamaGgufType.Int32   => LlamaGgufValue.Int32(  ReadInt32(s)),
        LlamaGgufType.Float32 => LlamaGgufValue.Float32(ReadFloat32(s)),
        LlamaGgufType.Bool    => LlamaGgufValue.Bool(   ReadUInt8(s) != 0),
        LlamaGgufType.String  => LlamaGgufValue.String( ReadGgufString(s)),
        LlamaGgufType.Uint64  => LlamaGgufValue.UInt64( ReadUInt64(s)),
        LlamaGgufType.Int64   => LlamaGgufValue.Int64(  ReadInt64(s)),
        LlamaGgufType.Float64 => LlamaGgufValue.Float64(ReadFloat64(s)),
        LlamaGgufType.Array   => ReadArrayValue(s),
        _ => throw new InvalidDataException($"Unknown gguf_type={type}.")
    };

    private static LlamaGgufValue ReadArrayValue(Stream s)
    {
        var inner = (LlamaGgufType)ReadUInt32(s);
        ulong count = ReadUInt64(s);
        if (count > int.MaxValue)
            throw new InvalidDataException($"Array of {count} elements exceeds .NET array limit.");

        if (inner == LlamaGgufType.String)
        {
            var arr = new string[count];
            for (ulong i = 0; i < count; i++) arr[i] = ReadGgufString(s);
            return LlamaGgufValue.StringArray(arr);
        }
        // Primitive arrays — read raw bytes then reinterpret as the typed array.
        return inner switch
        {
            LlamaGgufType.Uint8   => LlamaGgufValue.PrimitiveArray<byte>(  ReadPrimitiveArray<byte>(  s, (int)count)),
            LlamaGgufType.Int8    => LlamaGgufValue.PrimitiveArray<sbyte>( ReadPrimitiveArray<sbyte>( s, (int)count)),
            LlamaGgufType.Uint16  => LlamaGgufValue.PrimitiveArray<ushort>(ReadPrimitiveArray<ushort>(s, (int)count)),
            LlamaGgufType.Int16   => LlamaGgufValue.PrimitiveArray<short>( ReadPrimitiveArray<short>( s, (int)count)),
            LlamaGgufType.Uint32  => LlamaGgufValue.PrimitiveArray<uint>(  ReadPrimitiveArray<uint>(  s, (int)count)),
            LlamaGgufType.Int32   => LlamaGgufValue.PrimitiveArray<int>(   ReadPrimitiveArray<int>(   s, (int)count)),
            LlamaGgufType.Float32 => LlamaGgufValue.PrimitiveArray<float>( ReadPrimitiveArray<float>( s, (int)count)),
            LlamaGgufType.Bool    => LlamaGgufValue.PrimitiveArray<bool>(  ReadPrimitiveArray<bool>(  s, (int)count)),
            LlamaGgufType.Uint64  => LlamaGgufValue.PrimitiveArray<ulong>( ReadPrimitiveArray<ulong>( s, (int)count)),
            LlamaGgufType.Int64   => LlamaGgufValue.PrimitiveArray<long>(  ReadPrimitiveArray<long>(  s, (int)count)),
            LlamaGgufType.Float64 => LlamaGgufValue.PrimitiveArray<double>(ReadPrimitiveArray<double>(s, (int)count)),
            _ => throw new InvalidDataException($"Unsupported array inner type {inner}.")
        };
    }

    private static T[] ReadPrimitiveArray<T>(Stream s, int count) where T : unmanaged
    {
        var arr = new T[count];
        var bytes = MemoryMarshal.AsBytes(arr.AsSpan());
        ReadExactly(s, bytes);
        return arr;
    }

    private static string ReadGgufString(Stream s)
    {
        ulong len = ReadUInt64(s);
        if (len > int.MaxValue) throw new InvalidDataException($"GGUF string length {len} exceeds 2GB limit.");
        if (len == 0) return string.Empty;
        var buf = new byte[len];
        ReadExactly(s, buf);
        return Encoding.UTF8.GetString(buf);
    }

    private static byte   ReadUInt8(  Stream s) { Span<byte> b = stackalloc byte[1]; ReadExactly(s, b); return b[0]; }
    private static sbyte  ReadInt8(   Stream s) => unchecked((sbyte)ReadUInt8(s));
    private static ushort ReadUInt16( Stream s) { Span<byte> b = stackalloc byte[2]; ReadExactly(s, b); return BinaryPrimitives.ReadUInt16LittleEndian(b); }
    private static short  ReadInt16(  Stream s) { Span<byte> b = stackalloc byte[2]; ReadExactly(s, b); return BinaryPrimitives.ReadInt16LittleEndian(b); }
    private static uint   ReadUInt32( Stream s) { Span<byte> b = stackalloc byte[4]; ReadExactly(s, b); return BinaryPrimitives.ReadUInt32LittleEndian(b); }
    private static int    ReadInt32(  Stream s) { Span<byte> b = stackalloc byte[4]; ReadExactly(s, b); return BinaryPrimitives.ReadInt32LittleEndian(b); }
    private static float  ReadFloat32(Stream s) { Span<byte> b = stackalloc byte[4]; ReadExactly(s, b); return BinaryPrimitives.ReadSingleLittleEndian(b); }
    private static ulong  ReadUInt64( Stream s) { Span<byte> b = stackalloc byte[8]; ReadExactly(s, b); return BinaryPrimitives.ReadUInt64LittleEndian(b); }
    private static long   ReadInt64(  Stream s) { Span<byte> b = stackalloc byte[8]; ReadExactly(s, b); return BinaryPrimitives.ReadInt64LittleEndian(b); }
    private static double ReadFloat64(Stream s) { Span<byte> b = stackalloc byte[8]; ReadExactly(s, b); return BinaryPrimitives.ReadDoubleLittleEndian(b); }

    private static void ReadExactly(Stream s, Span<byte> buf)
    {
        int got = 0;
        while (got < buf.Length)
        {
            int n = s.Read(buf[got..]);
            if (n <= 0) throw new EndOfStreamException(
                $"GGUF read truncated: wanted {buf.Length} bytes, got {got}.");
            got += n;
        }
    }

    private static long AlignUp(long value, uint alignment) =>
        ((value + alignment - 1) / alignment) * alignment;

    /// <summary>
    /// Compute the on-disk byte size of a tensor with given <paramref name="typeId"/>
    /// and <paramref name="dims"/>. Mirrors <c>ggml_row_size</c> for the
    /// quant types we know about; for others (including any future
    /// additions) we throw rather than return a wrong size.
    /// </summary>
    private static long ComputeByteSize(uint typeId, long[] dims)
    {
        // Per-block size + block size for each ggml_type, copied from
        // ggml.c. Pairs of (bytes_per_block, elements_per_block).
        // Indexed by ggml_type integer; null entries mean "unknown to us".
        var typeBlock = TypeBlockSizes;
        if (typeId >= typeBlock.Length || typeBlock[typeId] is not var (bytesPerBlock, elementsPerBlock))
        {
            throw new InvalidDataException(
                $"Unknown tensor type {typeId} — extend LlamaGgufFile.TypeBlockSizes to support it.");
        }

        long elements = 1;
        foreach (var d in dims) elements *= d;
        if (elements % elementsPerBlock != 0)
        {
            throw new InvalidDataException(
                $"Tensor element count {elements} is not divisible by block size {elementsPerBlock} for type {typeId}.");
        }
        return (elements / elementsPerBlock) * bytesPerBlock;
    }

    /// <summary>Per-ggml_type block layout. Indexed by integer type id. Holds (bytes_per_block, elements_per_block).</summary>
    private static readonly (long bytesPerBlock, long elementsPerBlock)?[] TypeBlockSizes = BuildTypeBlockSizes();

    private static (long, long)?[] BuildTypeBlockSizes()
    {
        // 50-slot table covers everything ggml currently emits to GGUF
        // (highest in-use is NVFP4=40). Update if a future llama.cpp adds
        // a higher type id; the size assertion in ReadFromStream surfaces
        // unknown types rather than silently miscomputing.
        var t = new (long, long)?[50];

        // Plain numeric types — each "block" is a single element.
        t[0]  = (4, 1);   // F32
        t[1]  = (2, 1);   // F16
        t[24] = (1, 1);   // I8
        t[25] = (2, 1);   // I16
        t[26] = (4, 1);   // I32
        t[27] = (8, 1);   // I64
        t[28] = (8, 1);   // F64
        t[30] = (2, 1);   // BF16

        // Block-quant families — values from ggml.c's type_traits[].
        t[2]  = (18,  32);   // Q4_0
        t[3]  = (20,  32);   // Q4_1
        t[6]  = (22,  32);   // Q5_0
        t[7]  = (24,  32);   // Q5_1
        t[8]  = (34,  32);   // Q8_0
        t[9]  = (36,  32);   // Q8_1

        // K-quants — block size 256.
        t[10] = (84,  256);  // Q2_K (block_q2_K)
        t[11] = (110, 256);  // Q3_K
        t[12] = (144, 256);  // Q4_K
        t[13] = (176, 256);  // Q5_K
        t[14] = (210, 256);  // Q6_K
        t[15] = (292, 256);  // Q8_K

        // I-quants.
        t[16] = (66,  256);  // IQ2_XXS
        t[17] = (74,  256);  // IQ2_XS
        t[18] = (98,  256);  // IQ3_XXS
        t[19] = (50,  256);  // IQ1_S
        t[20] = (52,  32);   // IQ4_NL
        t[21] = (110, 256);  // IQ3_S
        t[22] = (82,  256);  // IQ2_S
        t[23] = (136, 256);  // IQ4_XS
        t[29] = (56,  256);  // IQ1_M

        // Ternary.
        t[34] = (54,  256);  // TQ1_0
        t[35] = (66,  256);  // TQ2_0

        // FP4 family.
        t[39] = (144, 256);  // MXFP4 (treated like Q4_K)
        t[40] = (144, 256);  // NVFP4 (placeholder — same shape as MXFP4)

        return t;
    }
}
