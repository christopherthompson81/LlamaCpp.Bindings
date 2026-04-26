using System.Buffers.Binary;
using System.Text.Json;

namespace LlamaCpp.Bindings;

/// <summary>
/// Element type in a safetensors file. Names match the on-disk
/// <c>"dtype"</c> string verbatim. We only enumerate the types HF
/// frameworks actually emit; quantized tensors land in HF as raw byte
/// buffers (not as a safetensors dtype) so they're outside the scope of
/// this reader.
/// </summary>
public enum LlamaSafetensorsDtype
{
    Unknown = 0,
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

/// <summary>One tensor entry in a safetensors file.</summary>
public sealed class LlamaSafetensorsTensor
{
    public string Name { get; }
    public LlamaSafetensorsDtype Dtype { get; }
    public long[] Shape { get; }

    /// <summary>Byte offset within the data section (NOT the file).</summary>
    public long ByteOffsetInDataSection { get; }
    public long ByteSize { get; }

    public LlamaSafetensorsTensor(string name, LlamaSafetensorsDtype dtype, long[] shape, long byteOffsetInDataSection, long byteSize)
    {
        Name = name;
        Dtype = dtype;
        Shape = shape;
        ByteOffsetInDataSection = byteOffsetInDataSection;
        ByteSize = byteSize;
    }

    /// <summary>Element count = product(<see cref="Shape"/>). Convenient for tensor-size sanity checks.</summary>
    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var d in Shape) n *= d;
            return n;
        }
    }

    /// <summary>Bytes per element for primitive dtypes. 0 if <see cref="Dtype"/> is unknown.</summary>
    public static int BytesPerElement(LlamaSafetensorsDtype d) => d switch
    {
        LlamaSafetensorsDtype.F64  => 8,
        LlamaSafetensorsDtype.F32  => 4,
        LlamaSafetensorsDtype.F16  => 2,
        LlamaSafetensorsDtype.BF16 => 2,
        LlamaSafetensorsDtype.I64  => 8,
        LlamaSafetensorsDtype.I32  => 4,
        LlamaSafetensorsDtype.I16  => 2,
        LlamaSafetensorsDtype.I8   => 1,
        LlamaSafetensorsDtype.U64  => 8,
        LlamaSafetensorsDtype.U32  => 4,
        LlamaSafetensorsDtype.U16  => 2,
        LlamaSafetensorsDtype.U8   => 1,
        LlamaSafetensorsDtype.Bool => 1,
        _ => 0,
    };
}

/// <summary>
/// Pure-C# reader for HuggingFace's safetensors format. Header layout:
/// <c>uint64 header_length + UTF-8 JSON header + tensor data section</c>.
/// The JSON header is parsed into managed objects on
/// <see cref="Open"/>; tensor bytes are streamed lazily from the
/// underlying file on <see cref="CopyTensorBytesAsync"/>.
/// </summary>
/// <remarks>
/// V1 supports single-file unsharded safetensors only — the convention
/// most small / medium HF checkpoints use. Sharded checkpoints
/// (<c>model-0000N-of-0000M.safetensors</c> + <c>model.safetensors.index.json</c>)
/// are deferred; adding them is mechanical (open each shard, route
/// tensors by name from the index).
/// </remarks>
public sealed class LlamaSafetensorsFile
{
    public string Path { get; }
    public long DataSectionFileOffset { get; }

    /// <summary>Tensor entries keyed by name. Lookup is ordinal.</summary>
    public IReadOnlyDictionary<string, LlamaSafetensorsTensor> Tensors { get; }

    /// <summary>Free-form file metadata from the <c>__metadata__</c> JSON entry. Empty when absent.</summary>
    public IReadOnlyDictionary<string, string> FileMetadata { get; }

    private LlamaSafetensorsFile(string path, long dataSectionOffset,
        IReadOnlyDictionary<string, LlamaSafetensorsTensor> tensors,
        IReadOnlyDictionary<string, string> fileMetadata)
    {
        Path = path;
        DataSectionFileOffset = dataSectionOffset;
        Tensors = tensors;
        FileMetadata = fileMetadata;
    }

    /// <summary>Open a safetensors file and read its JSON header.</summary>
    public static LlamaSafetensorsFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var fs = File.OpenRead(path);

        // 8-byte header length.
        Span<byte> lenBytes = stackalloc byte[8];
        ReadExactly(fs, lenBytes);
        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(lenBytes);

        // Sanity bound — real headers are well under 100 MB even for
        // huge MoE models. A garbage value here would otherwise allocate
        // unbounded memory before failing.
        if (headerLen > 100UL * 1024 * 1024)
        {
            throw new InvalidDataException(
                $"safetensors header length {headerLen} bytes is implausibly large; file is likely corrupt or non-safetensors.");
        }
        if (headerLen == 0)
        {
            throw new InvalidDataException("safetensors header length is 0 — empty / corrupt file.");
        }

        var headerBytes = new byte[headerLen];
        ReadExactly(fs, headerBytes);

        long dataSectionOffset = 8 + (long)headerLen;

        // Parse the JSON header. Each tensor entry has dtype/shape/data_offsets;
        // any "__metadata__" key is a string→string dict for free-form info.
        using var doc = JsonDocument.Parse(headerBytes);
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("safetensors header must be a JSON object.");
        }

        var tensors = new Dictionary<string, LlamaSafetensorsTensor>(StringComparer.Ordinal);
        var fileMetadata = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
            {
                if (prop.Value.ValueKind != JsonValueKind.Object) continue;
                foreach (var m in prop.Value.EnumerateObject())
                {
                    fileMetadata[m.Name] = m.Value.ValueKind == JsonValueKind.String
                        ? m.Value.GetString() ?? string.Empty
                        : m.Value.GetRawText();
                }
                continue;
            }

            // Tensor entry: { "dtype": "F32", "shape": [...], "data_offsets": [start, end] }
            if (prop.Value.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException(
                    $"Tensor '{prop.Name}' in safetensors header is not a JSON object.");
            }
            var entry = prop.Value;

            var dtype = ParseDtype(entry.GetProperty("dtype").GetString() ?? "");
            var shapeJson = entry.GetProperty("shape");
            var shape = new long[shapeJson.GetArrayLength()];
            for (int i = 0; i < shape.Length; i++) shape[i] = shapeJson[i].GetInt64();

            var offsets = entry.GetProperty("data_offsets");
            if (offsets.GetArrayLength() != 2)
            {
                throw new InvalidDataException(
                    $"Tensor '{prop.Name}' has data_offsets with {offsets.GetArrayLength()} elements; expected 2.");
            }
            long start = offsets[0].GetInt64();
            long end   = offsets[1].GetInt64();
            if (end < start || start < 0)
            {
                throw new InvalidDataException(
                    $"Tensor '{prop.Name}' has invalid data_offsets [{start}, {end}].");
            }
            tensors[prop.Name] = new LlamaSafetensorsTensor(
                prop.Name, dtype, shape, byteOffsetInDataSection: start, byteSize: end - start);
        }

        return new LlamaSafetensorsFile(path, dataSectionOffset, tensors, fileMetadata);
    }

    /// <summary>
    /// Read a tensor's raw bytes into a managed buffer. Suitable for
    /// small tensors only — large weight tensors should use
    /// <see cref="CopyTensorBytesToAsync"/> to avoid pulling them into
    /// managed memory.
    /// </summary>
    public byte[] ReadTensorBytes(string name)
    {
        if (!Tensors.TryGetValue(name, out var t))
            throw new KeyNotFoundException($"safetensors tensor '{name}' not found.");
        var bytes = new byte[t.ByteSize];
        using var fs = File.OpenRead(Path);
        fs.Seek(DataSectionFileOffset + t.ByteOffsetInDataSection, SeekOrigin.Begin);
        ReadExactly(fs, bytes);
        return bytes;
    }

    /// <summary>
    /// Stream a tensor's bytes into <paramref name="destination"/>.
    /// Used by the converter so weight tensors don't sit in managed
    /// memory while the rest of the GGUF is being assembled.
    /// </summary>
    public async Task CopyTensorBytesToAsync(string name, Stream destination, CancellationToken cancellationToken = default)
    {
        if (!Tensors.TryGetValue(name, out var t))
            throw new KeyNotFoundException($"safetensors tensor '{name}' not found.");
        await using var fs = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.Read);
        fs.Seek(DataSectionFileOffset + t.ByteOffsetInDataSection, SeekOrigin.Begin);

        var buf = new byte[Math.Min(t.ByteSize, 1 << 20)];
        long remaining = t.ByteSize;
        while (remaining > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int want = (int)Math.Min(buf.Length, remaining);
            int n = await fs.ReadAsync(buf.AsMemory(0, want), cancellationToken);
            if (n <= 0) throw new EndOfStreamException(
                $"safetensors data section truncated for tensor '{name}': " +
                $"{t.ByteSize - remaining}/{t.ByteSize} bytes copied.");
            await destination.WriteAsync(buf.AsMemory(0, n), cancellationToken);
            remaining -= n;
        }
    }

    /// <summary>Get a tensor entry by name; throws if absent.</summary>
    public LlamaSafetensorsTensor Get(string name)
    {
        if (!Tensors.TryGetValue(name, out var t))
            throw new KeyNotFoundException($"safetensors tensor '{name}' not found.");
        return t;
    }

    public bool Contains(string name) => Tensors.ContainsKey(name);

    private static LlamaSafetensorsDtype ParseDtype(string s) => s switch
    {
        "F64"  => LlamaSafetensorsDtype.F64,
        "F32"  => LlamaSafetensorsDtype.F32,
        "F16"  => LlamaSafetensorsDtype.F16,
        "BF16" => LlamaSafetensorsDtype.BF16,
        "I64"  => LlamaSafetensorsDtype.I64,
        "I32"  => LlamaSafetensorsDtype.I32,
        "I16"  => LlamaSafetensorsDtype.I16,
        "I8"   => LlamaSafetensorsDtype.I8,
        "U64"  => LlamaSafetensorsDtype.U64,
        "U32"  => LlamaSafetensorsDtype.U32,
        "U16"  => LlamaSafetensorsDtype.U16,
        "U8"   => LlamaSafetensorsDtype.U8,
        "BOOL" => LlamaSafetensorsDtype.Bool,
        _ => throw new InvalidDataException($"Unknown safetensors dtype '{s}'."),
    };

    private static void ReadExactly(Stream s, Span<byte> buf)
    {
        int got = 0;
        while (got < buf.Length)
        {
            int n = s.Read(buf[got..]);
            if (n <= 0) throw new EndOfStreamException(
                $"safetensors read truncated: wanted {buf.Length} bytes, got {got}.");
            got += n;
        }
    }
}
