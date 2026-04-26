using System.Text.Json;

namespace LlamaCpp.Bindings.HfConvert;

/// <summary>
/// Minimal wrapper around a HuggingFace <c>config.json</c>. Supports
/// dotted-path lookups (<c>"rope_scaling.factor"</c>) returning typed
/// values, plus a couple of convenience properties for fields the
/// converter engine always needs.
/// </summary>
public sealed class LlamaHfConfig : IDisposable
{
    /// <summary>Source directory (the parent of <c>config.json</c>). Used to locate sibling files.</summary>
    public string DirectoryPath { get; }

    /// <summary>The <c>"architectures"</c> array — used to pick a definition.</summary>
    public IReadOnlyList<string> ArchitectureNames { get; }

    /// <summary>Whether the model's lm_head shares weights with the embedding table.</summary>
    public bool TieWordEmbeddings { get; }

    /// <summary>The <c>"_name_or_path"</c> field, or <c>null</c>. Used as a fallback for general.name.</summary>
    public string? NameOrPath { get; }

    private readonly JsonDocument _doc;
    private bool _disposed;

    private LlamaHfConfig(string dir, JsonDocument doc, IReadOnlyList<string> archNames,
        bool tieWordEmbeddings, string? nameOrPath)
    {
        DirectoryPath = dir;
        _doc = doc;
        ArchitectureNames = archNames;
        TieWordEmbeddings = tieWordEmbeddings;
        NameOrPath = nameOrPath;
    }

    public static LlamaHfConfig FromDirectory(string dir)
    {
        ArgumentException.ThrowIfNullOrEmpty(dir);
        var path = Path.Combine(dir, "config.json");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"HF model directory '{dir}' is missing config.json.", path);
        }
        var json = File.ReadAllText(path);
        return FromJson(json, dir);
    }

    public static LlamaHfConfig FromJson(string json, string directoryPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(json);
        var doc = JsonDocument.Parse(json, new JsonDocumentOptions
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        });

        var archs = new List<string>();
        if (doc.RootElement.TryGetProperty("architectures", out var archNode)
            && archNode.ValueKind == JsonValueKind.Array)
        {
            foreach (var a in archNode.EnumerateArray())
            {
                if (a.ValueKind == JsonValueKind.String)
                {
                    archs.Add(a.GetString() ?? string.Empty);
                }
            }
        }

        bool tied = false;
        if (doc.RootElement.TryGetProperty("tie_word_embeddings", out var tieNode)
            && tieNode.ValueKind is JsonValueKind.True or JsonValueKind.False)
        {
            tied = tieNode.GetBoolean();
        }

        string? nameOrPath = null;
        if (doc.RootElement.TryGetProperty("_name_or_path", out var npNode)
            && npNode.ValueKind == JsonValueKind.String)
        {
            nameOrPath = npNode.GetString();
        }

        return new LlamaHfConfig(directoryPath, doc, archs, tied, nameOrPath);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _doc.Dispose();
    }

    /// <summary>
    /// Look up a value by dotted path. <c>"a.b.c"</c> walks
    /// <c>root["a"]["b"]["c"]</c>, returning <c>true</c> + the leaf
    /// element if every step exists, <c>false</c> otherwise.
    /// </summary>
    public bool TryGet(string dottedPath, out JsonElement value)
    {
        value = default;
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrEmpty(dottedPath)) return false;

        var current = _doc.RootElement;
        var span = dottedPath.AsSpan();
        while (!span.IsEmpty)
        {
            int dot = span.IndexOf('.');
            ReadOnlySpan<char> head = dot < 0 ? span : span[..dot];
            if (current.ValueKind != JsonValueKind.Object) return false;
            if (!current.TryGetProperty(head, out var next)) return false;
            current = next;
            span = dot < 0 ? ReadOnlySpan<char>.Empty : span[(dot + 1)..];
        }
        value = current;
        return true;
    }

    public uint?   GetUInt32(string p)  => TryGet(p, out var v) && v.TryGetUInt32(out var x) ? x : null;
    public int?    GetInt32(string p)   => TryGet(p, out var v) && v.TryGetInt32(out var x) ? x : null;
    public ulong?  GetUInt64(string p)  => TryGet(p, out var v) && v.TryGetUInt64(out var x) ? x : null;
    public long?   GetInt64(string p)   => TryGet(p, out var v) && v.TryGetInt64(out var x) ? x : null;
    public float?  GetFloat32(string p) => TryGet(p, out var v) && v.TryGetSingle(out var x) ? x : null;
    public double? GetFloat64(string p) => TryGet(p, out var v) && v.TryGetDouble(out var x) ? x : null;
    public bool?   GetBool(string p)    => TryGet(p, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
        ? v.GetBoolean() : null;
    public string? GetString(string p)  => TryGet(p, out var v) && v.ValueKind == JsonValueKind.String
        ? v.GetString() : null;
}
