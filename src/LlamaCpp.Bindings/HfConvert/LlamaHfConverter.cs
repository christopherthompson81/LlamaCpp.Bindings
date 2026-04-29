using System.Reflection;

namespace LlamaCpp.Bindings.HfConvert;

/// <summary>
/// Output element type for HF → GGUF conversion. F32 is lossless from
/// any HF source; F16 / BF16 are lossy when downcasting from F32 but
/// match the dtype of nearly every modern HF checkpoint
/// (BF16 is the dominant on-disk dtype since ~2023).
/// </summary>
public enum LlamaHfConvertOutputType : uint
{
    /// <summary>Full precision; doubles file size if source is F16/BF16.</summary>
    F32  = 0,
    /// <summary>IEEE half. Common for older Llama-2-era checkpoints.</summary>
    F16  = 1,
    /// <summary>Brain float. Default for most modern checkpoints — passthrough when source dtype matches.</summary>
    BF16 = 30,
}

/// <summary>Per-tensor progress reported during conversion.</summary>
public readonly record struct LlamaHfConvertProgress(
    int TensorIndex,
    int TensorCount,
    string CurrentTensorName);

/// <summary>Final summary returned by <see cref="LlamaHfConverter.ConvertAsync"/>.</summary>
public sealed record LlamaHfConvertResult(
    string OutputPath,
    string Architecture,
    int TensorCount,
    long OutputBytes,
    LlamaHfConvertOutputType OutputType,
    TimeSpan Elapsed);

/// <summary>
/// Engine that converts a HuggingFace model directory to a GGUF file,
/// driven by a <see cref="LlamaHfArchitectureDefinition"/>. The engine
/// is fully generic — adding a new architecture is normally a JSON
/// definition plus optionally a named transform; no engine code change
/// is required.
/// </summary>
/// <remarks>
/// <para>Inputs:</para>
/// <list type="bullet">
///   <item>HF model directory: <c>config.json</c>, <c>tokenizer.json</c>,
///         optionally <c>tokenizer_config.json</c>/<c>chat_template.json</c>,
///         and <c>model.safetensors</c> (V1: single-file unsharded).</item>
///   <item>Output GGUF path.</item>
///   <item>Output element type (F32/F16/BF16).</item>
/// </list>
/// <para>The engine reads the HF arch name from <c>config.json</c>, looks up
/// a matching definition, applies the metadata + tensor maps, runs the
/// tokenizer through <see cref="LlamaHfTokenizer"/>, and emits a GGUF
/// via <see cref="LlamaGgufWriter"/>.</para>
/// </remarks>
public static class LlamaHfConverter
{
    /// <summary>
    /// Convert <paramref name="hfDirectory"/> to a GGUF at
    /// <paramref name="outputPath"/>. Architecture is auto-detected
    /// from <c>config.json</c>'s "architectures" array.
    /// </summary>
    public static Task<LlamaHfConvertResult> ConvertAsync(
        string hfDirectory,
        string outputPath,
        LlamaHfConvertOutputType outputType = LlamaHfConvertOutputType.F16,
        IProgress<LlamaHfConvertProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(hfDirectory);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        return Task.Run(() => Convert(hfDirectory, outputPath, outputType, progress, cancellationToken), cancellationToken);
    }

    private static LlamaHfConvertResult Convert(
        string hfDirectory,
        string outputPath,
        LlamaHfConvertOutputType outputType,
        IProgress<LlamaHfConvertProgress>? progress,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        using var config = LlamaHfConfig.FromDirectory(hfDirectory);

        // Architecture detection: pick the first registered definition
        // whose hf_architectures intersects with the HF model's
        // "architectures" array.
        var def = SelectDefinition(config.ArchitectureNames)
            ?? throw new NotSupportedException(
                $"No architecture definition matches HF architectures {{{string.Join(", ", config.ArchitectureNames)}}}. " +
                $"Available:\n{FormatAvailableArchitectures()}");

        var safetensorsPath = Path.Combine(hfDirectory, "model.safetensors");
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException(
                $"HF model directory '{hfDirectory}' is missing model.safetensors. " +
                "V1 supports single-file unsharded safetensors only.",
                safetensorsPath);
        var safetensors = LlamaSafetensorsFile.Open(safetensorsPath);

        var tokenizer = LlamaHfTokenizer.FromDirectory(hfDirectory);

        var writer = new LlamaGgufWriter();

        // ----- general.* -----
        writer.SetMetadata("general.architecture", def.GgufArchitecture);
        writer.SetMetadata("general.name", DeriveModelName(hfDirectory, config));
        writer.SetMetadata("general.file_type", (uint)outputType);

        // ----- per-arch metadata (definition-driven) -----
        WriteArchitectureMetadata(writer, def, config);

        // ----- tokenizer.* -----
        // HF embeddings are commonly padded past the tokenizer's filled
        // entries to a multiple of 64 / 128 / 256 for GPU alignment. The
        // canonical convention (matching upstream convert_hf_to_gguf.py)
        // is that tokenizer.ggml.tokens has the same row count as the
        // embedding tensor, with [PAD_n] UNUSED entries filling the gap.
        // llama.cpp's tensor-shape check uses the tokens count as the
        // expected vocab dim, so the two must agree.
        int paddedVocabSize = Math.Max(
            tokenizer.Tokens.Count,
            ResolveVocabAnchor(def.VocabAnchor, config, safetensors, tokenizer.Tokens.Count));
        var tokenTypes = ApplyForceControlTokenPatterns(tokenizer, def);
        WriteTokenizerMetadata(writer, def, tokenizer, tokenTypes, paddedVocabSize);

        // ----- tensors -----
        var blockCount = ResolveBlockCount(def, config);
        var planned = PlanTensorWrites(def, blockCount, safetensors, config.TieWordEmbeddings);

        // Pre-resolve attention head counts once. Required by the
        // permute_q / permute_k transforms (Llama family); harmless to
        // resolve up-front for architectures that don't use them.
        // Both lookups are best-effort — architectures whose tensor map
        // never references permute_qk simply ignore the values.
        int? nHead   = TryResolveMetadataUInt32(def, config, "attention.head_count");
        int? nHeadKv = TryResolveMetadataUInt32(def, config, "attention.head_count_kv");

        for (int i = 0; i < planned.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var entry = planned[i];
            progress?.Report(new LlamaHfConvertProgress(i + 1, planned.Count, entry.GgufName));

            var srcInfo = safetensors.Get(entry.HfName);
            var srcBytes = safetensors.ReadTensorBytes(entry.HfName);

            // 1-D tensors (norms, biases) stay at F32 regardless of the
            // requested output type. ggml's compute path mixes them with
            // F32 activations and rejects an F16/BF16 norm with
            // "unsupported types: dst: f32, src0: f32, src1: f16". This
            // matches upstream convert_hf_to_gguf.py's heuristic.
            var effectiveOutputType = srcInfo.Shape.Length == 1
                ? LlamaHfConvertOutputType.F32
                : outputType;

            (byte[] outBytes, uint outTypeId) = entry.Mapping.Transform switch
            {
                "passthrough" => LlamaHfTensorTransforms.Passthrough(srcBytes, srcInfo.Dtype, effectiveOutputType),
                "permute_q"   => LlamaHfTensorTransforms.PermuteQK(
                                    srcBytes, srcInfo.Dtype, effectiveOutputType, srcInfo.Shape,
                                    nHead: nHead ?? throw new InvalidDataException(
                                        $"Tensor '{entry.HfName}' uses permute_q but architecture metadata lacks attention.head_count."),
                                    nHeadKv: nHead, isK: false),
                "permute_k"   => LlamaHfTensorTransforms.PermuteQK(
                                    srcBytes, srcInfo.Dtype, effectiveOutputType, srcInfo.Shape,
                                    nHead: nHead ?? throw new InvalidDataException(
                                        $"Tensor '{entry.HfName}' uses permute_k but architecture metadata lacks attention.head_count."),
                                    nHeadKv: nHeadKv, isK: true),
                _ => throw new NotSupportedException(
                    $"Tensor transform '{entry.Mapping.Transform}' is not in V1's library. Add it to LlamaHfTensorTransforms."),
            };

            // Reverse shape order: HF/PyTorch is row-major ([rows, cols] =
            // [vocab, hidden] for token_embd), ggml is fastest-varying-first
            // ([cols, rows] = [hidden, vocab]). The data bytes are identical
            // — only the shape metadata flips. Verified against the
            // canonical Qwen3-0.6B GGUF (token_embd is [1024, 151936]).
            var ggmlShape = new long[srcInfo.Shape.Length];
            for (int d = 0; d < srcInfo.Shape.Length; d++)
            {
                ggmlShape[d] = srcInfo.Shape[srcInfo.Shape.Length - 1 - d];
            }

            writer.AddTensor(entry.GgufName, outTypeId, ggmlShape, outBytes);
        }

        // ----- extra tensors (computed from config; not in safetensors) -----
        // Generators may return null when the config doesn't call for the
        // tensor (e.g., llama3 RoPE freqs are emitted only when
        // rope_scaling.rope_type == "llama3"). Skipped extras don't error.
        int extraCount = 0;
        foreach (var extra in def.ExtraTensors)
        {
            ct.ThrowIfCancellationRequested();
            var generated = extra.Generator switch
            {
                "llama3_rope_freqs" => LlamaHfExtraTensors.Llama3RopeFreqs(config),
                _ => throw new NotSupportedException(
                    $"Extra-tensor generator '{extra.Generator}' is not in V1's library. Add it to LlamaHfExtraTensors."),
            };
            if (generated is { } g)
            {
                writer.AddTensor(extra.Gguf, g.TypeId, g.Shape, g.Data);
                extraCount++;
            }
        }

        writer.WriteAsync(outputPath, ct).GetAwaiter().GetResult();
        sw.Stop();

        return new LlamaHfConvertResult(
            OutputPath: outputPath,
            Architecture: def.GgufArchitecture,
            TensorCount: planned.Count + extraCount,
            OutputBytes: new FileInfo(outputPath).Length,
            OutputType: outputType,
            Elapsed: sw.Elapsed);
    }

    private static string DeriveModelName(string hfDirectory, LlamaHfConfig config)
    {
        if (!string.IsNullOrEmpty(config.NameOrPath)) return config.NameOrPath!;
        try { return new DirectoryInfo(hfDirectory).Name; } catch { return "unknown"; }
    }

    private static int ResolveBlockCount(LlamaHfArchitectureDefinition def, LlamaHfConfig config)
    {
        // Find the metadata entry whose resolved gguf key is
        // "${arch}.block_count" — its HF path tells us where to look.
        // Required for tensor-template expansion of {i}.
        return TryResolveMetadataUInt32(def, config, "block_count")
            ?? throw new InvalidDataException(
                $"Architecture definition '{def.GgufArchitecture}' has no metadata entry mapping to '{def.GgufArchitecture}.block_count'.");
    }

    /// <summary>
    /// Best-effort resolve of a per-arch <c>uint32</c> metadata value:
    /// look up the metadata entry whose resolved gguf key matches
    /// <c>"&lt;gguf_arch&gt;.&lt;suffix&gt;"</c>, then read that field
    /// from <c>config.json</c>. Returns null when either the entry is
    /// missing from the definition or the field is missing from
    /// config.json (or isn't a uint32).
    /// </summary>
    private static int? TryResolveMetadataUInt32(
        LlamaHfArchitectureDefinition def, LlamaHfConfig config, string keySuffix)
    {
        var target = $"{def.GgufArchitecture}.{keySuffix}";
        foreach (var m in def.MetadataMap)
        {
            var resolvedGguf = m.Gguf.Replace("${arch}", def.GgufArchitecture, StringComparison.Ordinal);
            if (resolvedGguf == target)
            {
                var v = config.GetUInt32(m.Hf);
                if (v.HasValue) return (int)v.Value;
            }
        }
        return null;
    }

    private static void WriteArchitectureMetadata(
        LlamaGgufWriter writer, LlamaHfArchitectureDefinition def, LlamaHfConfig config)
    {
        foreach (var m in def.MetadataMap)
        {
            // ${arch} substitution: definitions write the FULL gguf key
            // with ${arch} as a placeholder (e.g. "${arch}.context_length"
            // or "general.author"). The substitution-token form lets a
            // single map handle both per-arch keys and global ones,
            // symmetric with {i} in tensor templates.
            var key = m.Gguf.Replace("${arch}", def.GgufArchitecture, StringComparison.Ordinal);
            if (!TryReadAndWrite(writer, key, m, config) && !m.Optional)
            {
                throw new InvalidDataException(
                    $"config.json is missing required field '{m.Hf}' for arch '{def.GgufArchitecture}' (gguf key '{key}').");
            }
        }
    }

    /// <summary>
    /// Resolve the configured <c>vocab_anchor</c> to an integer row count.
    /// </summary>
    /// <remarks>
    /// Format: <c>"config:&lt;dotted_path&gt;"</c> reads from
    /// <c>config.json</c>; <c>"tensor:&lt;name&gt;:dim&lt;N&gt;"</c> reads
    /// dimension N (0-indexed) of a safetensors tensor's shape. We fall
    /// back to <paramref name="fallback"/> if the anchor is missing —
    /// rather than throw — so a model with neither config.vocab_size nor
    /// the named tensor still converts (the engine just emits no
    /// padding, which is correct when there's nothing to pad to).
    /// </remarks>
    private static int ResolveVocabAnchor(
        string anchor, LlamaHfConfig config, LlamaSafetensorsFile safetensors, int fallback)
    {
        // Definition's init default is "config:vocab_size" — reflection-
        // based deserialization runs the field initializer, so anchor is
        // never empty for a well-formed definition. Defensive guard
        // anyway: a hand-edited definition could pass an empty string.
        if (string.IsNullOrEmpty(anchor)) return fallback;
        if (anchor.StartsWith("config:", StringComparison.Ordinal))
        {
            var path = anchor["config:".Length..];
            return (int)(config.GetUInt32(path) ?? (uint)fallback);
        }
        if (anchor.StartsWith("tensor:", StringComparison.Ordinal))
        {
            // Format: tensor:<name>:dim<N>
            var rest = anchor["tensor:".Length..];
            int sep = rest.LastIndexOf(":dim", StringComparison.Ordinal);
            if (sep < 0) return fallback;
            var name = rest[..sep];
            if (!int.TryParse(rest[(sep + ":dim".Length)..], out int dim)) return fallback;
            if (!safetensors.Contains(name)) return fallback;
            var shape = safetensors.Get(name).Shape;
            return dim >= 0 && dim < shape.Length ? (int)shape[dim] : fallback;
        }
        return fallback;
    }

    /// <summary>
    /// Build a token-types array that copies <see cref="LlamaHfTokenizer.TokenTypes"/>
    /// and overrides matching entries to <see cref="LlamaTokenTypeId.Control"/>.
    /// Returning a fresh array keeps the tokenizer immutable from the
    /// engine's perspective.
    /// </summary>
    private static int[] ApplyForceControlTokenPatterns(LlamaHfTokenizer tokenizer, LlamaHfArchitectureDefinition def)
    {
        var types = new int[tokenizer.TokenTypes.Count];
        for (int i = 0; i < types.Length; i++) types[i] = tokenizer.TokenTypes[i];

        var patterns = def.ForceControlTokenPatterns
            ?? LlamaHfArchitectureDefinition.DefaultControlTokenPatterns;
        if (patterns.Count == 0) return types;

        for (int i = 0; i < tokenizer.Tokens.Count; i++)
        {
            var tok = tokenizer.Tokens[i];
            // Cheap fast-path: only check tokens that look at all
            // control-shaped (start with '<' or '['). Most BPE vocab
            // entries are plain word fragments and skip the matcher.
            if (tok.Length == 0 || (tok[0] != '<' && tok[0] != '[')) continue;
            foreach (var p in patterns)
            {
                if (GlobMatches(tok, p))
                {
                    types[i] = LlamaTokenTypeId.Control;
                    break;
                }
            }
        }
        return types;
    }

    /// <summary>
    /// Simple glob match supporting <c>*</c> and <c>?</c> wildcards.
    /// Uses <see cref="System.IO.Enumeration.FileSystemName.MatchesSimpleExpression"/>
    /// — battle-tested glob semantics shared with the file-system APIs.
    /// </summary>
    private static bool GlobMatches(string value, string pattern) =>
        System.IO.Enumeration.FileSystemName.MatchesSimpleExpression(pattern, value);

    private static bool TryReadAndWrite(LlamaGgufWriter writer, string ggufKey, MetadataMappingEntry m, LlamaHfConfig config)
    {
        switch (m.Type)
        {
            case "uint32":
                if (config.GetUInt32(m.Hf) is uint u32) { writer.SetMetadata(ggufKey, u32); return true; }
                return false;
            case "int32":
                if (config.GetInt32(m.Hf) is int i32) { writer.SetMetadata(ggufKey, i32); return true; }
                return false;
            case "uint64":
                if (config.GetUInt64(m.Hf) is ulong u64) { writer.SetMetadata(ggufKey, u64); return true; }
                return false;
            case "int64":
                if (config.GetInt64(m.Hf) is long i64) { writer.SetMetadata(ggufKey, i64); return true; }
                return false;
            case "float32":
                if (config.GetFloat32(m.Hf) is float f32) { writer.SetMetadata(ggufKey, f32); return true; }
                return false;
            case "float64":
                if (config.GetFloat64(m.Hf) is double f64) { writer.SetMetadata(ggufKey, f64); return true; }
                return false;
            case "bool":
                if (config.GetBool(m.Hf) is bool b) { writer.SetMetadata(ggufKey, b); return true; }
                return false;
            case "string":
                var s = config.GetString(m.Hf);
                if (s is not null) { writer.SetMetadata(ggufKey, s); return true; }
                return false;
            default:
                throw new InvalidDataException(
                    $"Unsupported metadata type '{m.Type}' (gguf_key='{ggufKey}').");
        }
    }

    private static void WriteTokenizerMetadata(
        LlamaGgufWriter writer, LlamaHfArchitectureDefinition def, LlamaHfTokenizer tokenizer,
        int[] tokenTypes, int paddedVocabSize)
    {
        // V1 supports only "bpe-gpt2" tokenizer family. Architecture
        // definitions can declare other families; we'll add readers as
        // they're needed.
        if (def.TokenizerFamily != "bpe-gpt2")
        {
            throw new NotSupportedException(
                $"Tokenizer family '{def.TokenizerFamily}' is not in V1. Only 'bpe-gpt2' is supported.");
        }

        writer.SetMetadata("tokenizer.ggml.model", "gpt2");
        if (!string.IsNullOrEmpty(def.TokenizerPre))
        {
            writer.SetMetadata("tokenizer.ggml.pre", def.TokenizerPre);
        }

        // Pad tokens + token_types up to paddedVocabSize so they match
        // the embedding tensor's row count. Token types come from the
        // possibly-overridden array passed in by the caller.
        int n = Math.Max(tokenizer.Tokens.Count, paddedVocabSize);
        var paddedTokens = new string[n];
        var paddedTypes = new int[n];
        for (int i = 0; i < tokenizer.Tokens.Count; i++)
        {
            paddedTokens[i] = tokenizer.Tokens[i];
            paddedTypes[i]  = tokenTypes[i];
        }
        for (int i = tokenizer.Tokens.Count; i < n; i++)
        {
            paddedTokens[i] = $"[PAD{i}]";
            paddedTypes[i]  = LlamaTokenTypeId.Unused;
        }

        writer.SetMetadata("tokenizer.ggml.tokens", LlamaGgufValue.StringArray(paddedTokens));
        writer.SetMetadata("tokenizer.ggml.token_type",
            LlamaGgufValue.PrimitiveArray<int>(paddedTypes));
        if (tokenizer.Merges.Count > 0)
        {
            writer.SetMetadata("tokenizer.ggml.merges", LlamaGgufValue.StringArray(tokenizer.Merges));
        }
        foreach (var (suffix, id) in tokenizer.SpecialTokenIds)
        {
            writer.SetMetadata($"tokenizer.ggml.{suffix}", id);
        }

        // End-of-turn token: resolve the configured string against the
        // padded vocab and emit its id. llama.cpp adds eot to its
        // end-of-generation set, so chat models stop at turn boundaries.
        if (!string.IsNullOrEmpty(def.EotToken))
        {
            int eotId = -1;
            for (int i = 0; i < paddedTokens.Length; i++)
            {
                if (string.Equals(paddedTokens[i], def.EotToken, StringComparison.Ordinal))
                {
                    eotId = i;
                    break;
                }
            }
            if (eotId >= 0)
            {
                writer.SetMetadata("tokenizer.ggml.eot_token_id", (uint)eotId);
            }
            // If the configured eot string isn't in the vocab we don't
            // throw — it's an architecture-definition hint, not a hard
            // requirement, and a missing eot at worst restores the
            // pre-fix behaviour where llama.cpp warns at load.
        }

        if (!string.IsNullOrEmpty(tokenizer.ChatTemplate))
        {
            writer.SetMetadata("tokenizer.chat_template", tokenizer.ChatTemplate);
        }
    }

    /// <summary>
    /// Expand the tensor map's <c>{i}</c> templates over [0, blockCount)
    /// and resolve which entries actually exist in the safetensors,
    /// honoring <c>optional</c> + <c>optional_when_tied</c> flags.
    /// </summary>
    private static List<PlannedTensor> PlanTensorWrites(
        LlamaHfArchitectureDefinition def, int blockCount,
        LlamaSafetensorsFile safetensors, bool tieWordEmbeddings)
    {
        var planned = new List<PlannedTensor>();
        foreach (var entry in def.TensorMap)
        {
            bool isPerBlock = entry.Hf.Contains("{i}", StringComparison.Ordinal)
                              || entry.Gguf.Contains("{i}", StringComparison.Ordinal);
            if (isPerBlock)
            {
                for (int i = 0; i < blockCount; i++)
                {
                    PlanOne(planned, entry, safetensors, tieWordEmbeddings,
                        i.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
            }
            else
            {
                PlanOne(planned, entry, safetensors, tieWordEmbeddings, blockIndex: null);
            }
        }
        return planned;
    }

    private static void PlanOne(
        List<PlannedTensor> dst, TensorMappingEntry entry,
        LlamaSafetensorsFile safetensors, bool tieWordEmbeddings, string? blockIndex)
    {
        string hf = blockIndex is null ? entry.Hf : entry.Hf.Replace("{i}", blockIndex, StringComparison.Ordinal);
        string gguf = blockIndex is null ? entry.Gguf : entry.Gguf.Replace("{i}", blockIndex, StringComparison.Ordinal);

        if (!safetensors.Contains(hf))
        {
            // Tied embeddings: skip output.weight if the model declares it.
            if (entry.OptionalWhenTied && tieWordEmbeddings) return;
            if (entry.Optional) return;
            throw new InvalidDataException(
                $"Required HF tensor '{hf}' (→ GGUF '{gguf}') not found in safetensors. " +
                $"Tie-word-embeddings={tieWordEmbeddings}.");
        }
        dst.Add(new PlannedTensor(hf, gguf, entry));
    }

    private sealed record PlannedTensor(string HfName, string GgufName, TensorMappingEntry Mapping);

    // ----- Definition registry -----

    private static readonly Lazy<List<LlamaHfArchitectureDefinition>> _definitions = new(LoadEmbeddedDefinitions);

    /// <summary>
    /// Pick the first registered definition whose primary
    /// <c>hf_architecture</c> or any of its aliases matches one of
    /// <paramref name="hfArchitectureNames"/>.
    /// </summary>
    public static LlamaHfArchitectureDefinition? SelectDefinition(IReadOnlyList<string> hfArchitectureNames)
    {
        foreach (var def in _definitions.Value)
        {
            foreach (var hf in def.AllHfArchitectures)
            {
                foreach (var query in hfArchitectureNames)
                {
                    if (string.Equals(hf, query, StringComparison.Ordinal)) return def;
                }
            }
        }
        return null;
    }

    /// <summary>List the GGUF architecture names this build can convert.</summary>
    public static IReadOnlyList<string> AvailableArchitectures() =>
        _definitions.Value.Select(d => d.GgufArchitecture).ToArray();

    /// <summary>
    /// Human-readable list of available architectures with their
    /// definition descriptions, suitable for error messages and the
    /// GUI's "supported architectures" affordance. This is the consumer
    /// of the <c>description</c> field in architecture JSONs.
    /// </summary>
    public static string FormatAvailableArchitectures()
    {
        var sb = new System.Text.StringBuilder();
        foreach (var def in _definitions.Value)
        {
            sb.Append("  - ").Append(def.GgufArchitecture);
            if (!string.IsNullOrEmpty(def.Description))
            {
                sb.Append(": ").Append(def.Description);
            }
            sb.Append(" (HF: ").Append(string.Join(", ", def.AllHfArchitectures)).AppendLine(")");
        }
        return sb.ToString();
    }

    private static List<LlamaHfArchitectureDefinition> LoadEmbeddedDefinitions()
    {
        var defs = new List<LlamaHfArchitectureDefinition>();
        var asm = typeof(LlamaHfConverter).Assembly;
        foreach (var name in asm.GetManifestResourceNames())
        {
            if (!name.StartsWith("LlamaCpp.Bindings.HfConvert.architectures.", StringComparison.Ordinal)) continue;
            if (!name.EndsWith(".json", StringComparison.Ordinal)) continue;
            using var s = asm.GetManifestResourceStream(name);
            if (s is null) continue;
            using var reader = new StreamReader(s);
            var json = reader.ReadToEnd();
            try
            {
                defs.Add(LlamaHfArchitectureDefinition.FromJson(json));
            }
            catch (Exception ex)
            {
                throw new InvalidDataException(
                    $"Embedded architecture definition '{name}' is malformed: {ex.Message}", ex);
            }
        }
        return defs;
    }
}
