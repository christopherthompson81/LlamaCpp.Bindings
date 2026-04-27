using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings;

/// <summary>
/// Orchestrates the per-category PPL ablation campaign that produces a
/// <see cref="LlamaSensitivityProfile"/>. Conceptually:
/// <list type="number">
///   <item>Quantize the source GGUF to a temp F16 baseline.</item>
///   <item>For every (category, candidate type) pair: build a recipe
///         that pins this category's tensors to the candidate type and
///         everything else to F16, then quantize.</item>
///   <item>Run all (1 + N×M) PPL passes through
///         <see cref="LlamaPerplexity.RunParallelAsync"/> at the
///         configured concurrency — this is the parallel unlock that
///         makes profile-building tractable (<c>~6 min</c> for
///         Qwen3-0.6B on a 3090 + i7-10700K vs <c>~88 min</c> serial).</item>
///   <item>Compute ΔPPL = ablation_ppl − baseline_ppl per cell.</item>
///   <item>Detect each category's "knee" — the smallest type before
///         ΔPPL crosses a catastrophic threshold (default 5.0 PPL).
///         Recipes built from this profile won't drop a category below
///         its knee.</item>
/// </list>
/// </summary>
public static class LlamaSensitivityProfileBuilder
{
    /// <summary>Tunable knobs for <see cref="BuildAsync"/>.</summary>
    public sealed class Options
    {
        /// <summary>
        /// Candidate types to ablate at. Default {Q2_K, Q4_K, Q6_K} mirrors
        /// the Stage-2 ablation campaign and gives enough resolution to
        /// detect knees while staying inside ~22 PPL runs for a 7-category
        /// model.
        /// </summary>
        public IReadOnlyList<LlamaTensorType> CandidateTypes { get; set; } =
            new[] { LlamaTensorType.Q2_K, LlamaTensorType.Q4_K, LlamaTensorType.Q6_K };

        /// <summary>
        /// Categories to score. Default covers the 7 weight categories of
        /// a standard transformer; profiles stay comparable across
        /// architectures even though the *coefficients* differ.
        /// </summary>
        public IReadOnlyList<string> Categories { get; set; } = new[]
        {
            "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_output.weight", "ffn_up", "ffn_gate", "ffn_down",
        };

        /// <summary>Optional imatrix GGUF path to use for imatrix-aware quantization. Recommended.</summary>
        public string? ImatrixPath { get; set; }

        /// <summary>
        /// Concurrency cap for the inner PPL runner. <c>0</c> (default)
        /// means "auto" — <see cref="LlamaPerplexity.RecommendConcurrency"/>
        /// looks at the actual ablation file sizes and the configured VRAM
        /// budget to pick a value that scales with model size. The bench-
        /// derived ceiling on 16-logical-core CPUs is 8 (softmax wants
        /// 2+ threads/job); on ~24 GB VRAM the per-instance budget caps
        /// it at 8 / 5 / 2 for 0.6B / 1.7B / 4B-class F16 sources.
        /// </summary>
        public int MaxConcurrent { get; set; } = 0;

        /// <summary>
        /// Override available GPU VRAM in bytes (default: 24 GB / RTX 3090
        /// class). Pass smaller for tighter hardware. Forwards to
        /// <see cref="LlamaPerplexity.RecommendConcurrency"/>.
        /// </summary>
        public long? AvailableVramBytes { get; set; }

        /// <summary>
        /// Working directory for temp quantized files. Set to a fast disk
        /// — the campaign produces <c>1 + categories × candidates</c>
        /// quantized variants of the input. Defaults to a fresh tempdir.
        /// </summary>
        public string? WorkingDirectory { get; set; }

        /// <summary>If true, delete the temp quants when the campaign finishes.</summary>
        public bool CleanupWorkingDirectory { get; set; } = true;

        /// <summary>
        /// PPL options passed through to every inner perplexity run.
        /// Defaults match GGUFLab's other perplexity tools (n_ctx=512,
        /// score-second-half-only — matches llama.cpp's published numbers).
        /// </summary>
        public LlamaPerplexityOptions? PerplexityOptions { get; set; }

        /// <summary>
        /// Path to a JSON checkpoint that survives builder crashes. After
        /// each PPL completes its result is appended and the file is
        /// atomically rewritten; on a subsequent run the builder loads
        /// it, validates it matches this campaign's source/corpus/specs,
        /// and skips any (category, type) already scored. Default
        /// (null) places <c>checkpoint.json</c> inside the working
        /// directory so resumability is on by default — the cost is one
        /// small file write per completed PPL. Set to a path outside
        /// <see cref="WorkingDirectory"/> if you want the checkpoint to
        /// outlive <see cref="CleanupWorkingDirectory"/>.
        /// </summary>
        public string? CheckpointPath { get; set; }

        /// <summary>
        /// Per-category catastrophic threshold used to compute
        /// <see cref="LlamaSensitivityCategoryCoefficient.RecommendedFloor"/>.
        /// A type is below the floor if its ΔPPL exceeds this value;
        /// recipes won't choose below the floor. 5.0 is a generous
        /// default — picks up genuine knees (Run 11 saw 124 PPL on
        /// `ffn_down` at Q2_K) without flagging mild degradation as a
        /// knee.
        /// </summary>
        public double KneeDeltaPplThreshold { get; set; } = 5.0;
    }

    /// <summary>Progress event raised as the campaign advances.</summary>
    public sealed record Progress(
        Stage Stage,
        int CompletedJobs,
        int TotalJobs,
        string? CurrentLabel = null);

    /// <summary>Coarse phases of the campaign for progress reporting.</summary>
    public enum Stage
    {
        Quantizing,
        Scoring,
        Done,
    }

    /// <summary>Build a sensitivity profile for <paramref name="sourceModelPath"/>.</summary>
    public static async Task<LlamaSensitivityProfile> BuildAsync(
        string sourceModelPath,
        string corpusPath,
        Options? options = null,
        IProgress<Progress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourceModelPath);
        ArgumentException.ThrowIfNullOrEmpty(corpusPath);
        if (!File.Exists(sourceModelPath))
            throw new FileNotFoundException($"Source model not found: {sourceModelPath}", sourceModelPath);
        if (!File.Exists(corpusPath))
            throw new FileNotFoundException($"Corpus not found: {corpusPath}", corpusPath);

        var opts = options ?? new Options();
        var corpusText = await File.ReadAllTextAsync(corpusPath, cancellationToken).ConfigureAwait(false);

        var workDir = opts.WorkingDirectory ?? Path.Combine(
            Path.GetTempPath(),
            "llama-sensitivity-profile-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workDir);

        try
        {
            // --- 1. Enumerate the source's 2-D weight tensors and group
            //        them by category so we can build per-category recipes.
            var ggufFile = LlamaGgufFile.Open(sourceModelPath);
            var architectureId = ResolveArchitecture(ggufFile);
            var layerCount = ResolveLayerCount(ggufFile);

            var weightTensors = ggufFile.Tensors
                .Where(t => t.Dimensions.Length > 1 && t.Name.EndsWith(".weight", StringComparison.Ordinal))
                .Select(t => t.Name)
                .ToList();

            // Validate every requested category has at least one tensor —
            // catches typos and architectures missing a category early.
            foreach (var c in opts.Categories)
            {
                if (!weightTensors.Any(n => CategoryMatch(n, c)))
                {
                    throw new InvalidOperationException(
                        $"Category '{c}' has no matching tensors in {sourceModelPath} — typo, or this " +
                        "architecture doesn't have that category. Adjust Options.Categories.");
                }
            }

            // --- 2. Build the F16 baseline + every (category, type) ablation
            //        quant. All written into workDir as <label>.gguf.
            var baselinePath = Path.Combine(workDir, "baseline.gguf");
            int totalJobs = 1 + opts.Categories.Count * opts.CandidateTypes.Count;
            int completed = 0;

            void ReportQuant(string label) =>
                progress?.Report(new Progress(Stage.Quantizing, completed, totalJobs, label));

            // Resolve concurrency early so we can size the disk-bounded
            // batches. We use the F16 source as the size estimate — every
            // ablation file is roughly the same size as the source (it's
            // mostly F16 with one category dropped to a smaller type).
            int totalAblations = opts.Categories.Count * opts.CandidateTypes.Count;
            int resolvedConcurrency = opts.MaxConcurrent > 0
                ? opts.MaxConcurrent
                : LlamaPerplexity.RecommendConcurrency(
                    new[] { sourceModelPath },
                    availableVramBytes: opts.AvailableVramBytes,
                    expectedJobCount: totalAblations);

            // Default to n_ctx=512 to match the wikitext-2 published-number
            // convention used in Run 9/11 (and what GGUFLab's standalone
            // Perplexity tool uses). The bindings' raw default is 2048,
            // which gives lower absolute PPL but doesn't match the
            // existing investigation numbers.
            var pplOpts = opts.PerplexityOptions ?? new LlamaPerplexityOptions { ContextSize = 512 };
            var ablationPpl = new Dictionary<(string Cat, LlamaTensorType Type), double>();

            // ---- Resumability ----
            // Load any prior checkpoint that was written by a crashed or
            // interrupted earlier run. We only honor it if the campaign
            // signature matches (same source, same corpus, same specs);
            // otherwise we ignore it and start fresh — better than
            // silently merging partial results from a different setup.
            var checkpointPath = opts.CheckpointPath ?? Path.Combine(workDir, "checkpoint.json");
            var corpusName = Path.GetFileName(corpusPath);
            var checkpoint = TryLoadCheckpoint(checkpointPath, sourceModelPath, corpusName,
                opts.Categories, opts.CandidateTypes);
            double? resumedBaseline = checkpoint?.Baseline;
            if (checkpoint is not null)
            {
                foreach (var (key, ppl) in checkpoint.Ablations)
                {
                    var split = key.Split('|');
                    if (split.Length == 2 && Enum.TryParse<LlamaTensorType>(split[1], out var t))
                        ablationPpl[(split[0], t)] = ppl;
                }
            }

            // ---- Pipelined disk-bounded campaign ----
            // The naive design ("quantize all 22 files, then PPL them all,
            // then delete") needs ~22× the source size on disk. For
            // Qwen3-1.7B F16 that's 77 GB, which busts /tmp on most
            // machines. Instead we pipeline in batches of `concurrency`:
            //   • Quantize the next batch (sequential — quantize is fast)
            //   • Run those PPLs through the parallel runner
            //   • Delete each ablation's file as soon as its PPL completes
            // Disk peak is bounded by concurrency × file_size — so 8 × 1.4 GB
            // = ~11 GB for the 0.6B model, 5 × 3.5 GB = ~18 GB for 1.7B.

            // First: baseline. One file in flight, then deleted. If the
            // checkpoint already has it, skip both the quantize and the
            // PPL — saves ~30 s on small models, more on big ones.
            double baseline;
            int scored;
            if (resumedBaseline is double cachedBaseline)
            {
                baseline = cachedBaseline;
                completed++;  // count the would-have-quantized baseline
                scored = 1 + ablationPpl.Count;  // baseline + any cached ablations
                progress?.Report(new Progress(Stage.Scoring, scored, totalJobs,
                    CurrentLabel: $"resumed from checkpoint (baseline + {ablationPpl.Count} ablations)"));
            }
            else
            {
                ReportQuant("baseline (F16)");
                await QuantizeAsync(sourceModelPath, baselinePath,
                    ftype: LlamaFileType.MostlyF16,
                    imatrixPath: opts.ImatrixPath,
                    recipe: null,
                    cancellationToken).ConfigureAwait(false);
                completed++;

                progress?.Report(new Progress(Stage.Scoring, 0, totalJobs,
                    CurrentLabel: $"baseline (concurrency={resolvedConcurrency})"));
                var baselineJob = new LlamaPerplexity.PerplexityJob(
                    ModelPath: baselinePath, Corpus: corpusText, Options: pplOpts, Tag: "BASELINE");
                baseline = double.NaN;
                await foreach (var jr in LlamaPerplexity.RunParallelAsync(
                    new[] { baselineJob }, maxConcurrent: 1, cancellationToken: cancellationToken))
                {
                    baseline = jr.Result.Perplexity;
                }
                try { File.Delete(baselinePath); } catch { /* best-effort */ }
                scored = 1;
                SaveCheckpoint(checkpointPath, sourceModelPath, corpusName,
                    opts.Categories, opts.CandidateTypes, baseline, ablationPpl);
                progress?.Report(new Progress(Stage.Scoring, scored, totalJobs, CurrentLabel: "baseline done"));
            }

            // Now the ablation specs, batched into rounds of `resolvedConcurrency`.
            // Skip any (cat, type) the checkpoint already has scored.
            var allSpecs = new List<(string Category, LlamaTensorType Type)>();
            foreach (var category in opts.Categories)
                foreach (var type in opts.CandidateTypes)
                    if (!ablationPpl.ContainsKey((category, type)))
                        allSpecs.Add((category, type));
            // Account for already-completed jobs in `completed` so progress
            // reports remain meaningful (denominator is total, not remaining).
            completed += ablationPpl.Count;

            for (int batchStart = 0; batchStart < allSpecs.Count; batchStart += resolvedConcurrency)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var batch = allSpecs.Skip(batchStart).Take(resolvedConcurrency).ToList();
                var batchPaths = new List<(string Cat, LlamaTensorType Type, string Path)>();

                // Quantize this batch sequentially (quantize is CPU-only
                // and not the bottleneck — overlapping with PPL would
                // double the disk peak for marginal time savings).
                foreach (var (cat, type) in batch)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var slug = $"{Slugify(cat)}_{type}";
                    var outPath = Path.Combine(workDir, $"{slug}.gguf");
                    ReportQuant($"{cat} @ {type}");
                    var recipe = BuildAblationRecipe(weightTensors, cat, type);
                    await QuantizeAsync(sourceModelPath, outPath,
                        ftype: LlamaFileType.Q4_K_M,
                        imatrixPath: opts.ImatrixPath,
                        recipe: recipe,
                        cancellationToken).ConfigureAwait(false);
                    batchPaths.Add((cat, type, outPath));
                    completed++;
                }

                // Run PPLs for this batch in parallel. As each finishes,
                // delete its file to keep disk peak bounded.
                var batchJobs = batchPaths.Select(bp =>
                    new LlamaPerplexity.PerplexityJob(
                        ModelPath: bp.Path, Corpus: corpusText, Options: pplOpts,
                        Tag: $"{bp.Cat}|{bp.Type}|{bp.Path}")).ToList();

                await foreach (var jr in LlamaPerplexity.RunParallelAsync(
                    batchJobs, resolvedConcurrency, cancellationToken: cancellationToken))
                {
                    scored++;
                    if (jr.Tag is string tag)
                    {
                        var split = tag.Split('|');
                        var cat = split[0];
                        var type = Enum.Parse<LlamaTensorType>(split[1]);
                        var path = split[2];
                        ablationPpl[(cat, type)] = jr.Result.Perplexity;
                        SaveCheckpoint(checkpointPath, sourceModelPath, corpusName,
                            opts.Categories, opts.CandidateTypes, baseline, ablationPpl);
                        progress?.Report(new Progress(Stage.Scoring, scored, totalJobs,
                            CurrentLabel: $"{cat} @ {type} = {jr.Result.Perplexity:F4}"));
                        try { File.Delete(path); } catch { /* best-effort */ }
                    }
                }
            }

            if (double.IsNaN(baseline))
                throw new InvalidOperationException("Baseline PPL never returned from runner.");

            // --- 4. Build the per-category coefficient records.
            var categories = new Dictionary<string, LlamaSensitivityCategoryCoefficient>();
            foreach (var cat in opts.Categories)
            {
                var deltas = new Dictionary<LlamaTensorType, double>();
                foreach (var type in opts.CandidateTypes)
                {
                    if (ablationPpl.TryGetValue((cat, type), out var ppl))
                        deltas[type] = ppl - baseline;
                }

                LlamaTensorType? floor = null;
                // Walk types small→large; first one whose ΔPPL is below the
                // catastrophic threshold sets the floor (anything smaller
                // than that is the "do not cross" band).
                foreach (var type in opts.CandidateTypes
                             .OrderBy(t => GetBitsPerElement(t)))
                {
                    if (deltas.TryGetValue(type, out var d) && d <= opts.KneeDeltaPplThreshold)
                    {
                        floor = type;
                        break;
                    }
                }
                categories[cat] = new LlamaSensitivityCategoryCoefficient(deltas, floor);
            }

            progress?.Report(new Progress(Stage.Done, scored, totalJobs));
            var provenance = new LlamaSensitivityProvenance(
                Method:               "ablation",
                SourceModel:          Path.GetFileName(sourceModelPath),
                SourceParameterCount: ResolveParameterCount(ggufFile),
                Corpus:               Path.GetFileName(corpusPath),
                BuiltAtUtc:           DateTime.UtcNow,
                BuilderVersion:       BuilderVersionString);
            return new LlamaSensitivityProfile(
                SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
                ArchitectureId:        architectureId,
                LayerCount:            layerCount,
                FamilyNotes:           null,
                Provenance:            provenance,
                F16BaselinePerplexity: baseline,
                BaselineContextSize:   pplOpts.ContextSize,
                Categories:            categories);
        }
        finally
        {
            if (opts.CleanupWorkingDirectory)
            {
                try { Directory.Delete(workDir, recursive: true); }
                catch { /* best-effort */ }
            }
        }
    }

    // ---- checkpoint -----------------------------------------------------

    /// <summary>
    /// On-disk checkpoint shape. Carries enough identity bits
    /// (<c>SourceModelPath</c>, <c>CorpusName</c>, <c>Categories</c>,
    /// <c>CandidateTypes</c>) to detect mismatch and refuse to resume
    /// across incompatible campaigns.
    /// </summary>
    private sealed record Checkpoint(
        string SourceModelPath,
        string CorpusName,
        IReadOnlyList<string> Categories,
        IReadOnlyList<LlamaTensorType> CandidateTypes,
        double? Baseline,
        Dictionary<string, double> Ablations,
        DateTime UpdatedAtUtc);

    private static readonly JsonSerializerOptions CheckpointJsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    private static Checkpoint? TryLoadCheckpoint(
        string path, string sourceModelPath, string corpusName,
        IReadOnlyList<string> categories, IReadOnlyList<LlamaTensorType> candidateTypes)
    {
        if (!File.Exists(path)) return null;
        try
        {
            var cp = JsonSerializer.Deserialize<Checkpoint>(File.ReadAllText(path), CheckpointJsonOpts);
            if (cp is null) return null;
            // Reject mismatched campaigns — silently merging would produce
            // a bogus profile (different corpus → different baselines).
            if (cp.SourceModelPath != sourceModelPath) return null;
            if (cp.CorpusName != corpusName) return null;
            if (!cp.Categories.SequenceEqual(categories)) return null;
            if (!cp.CandidateTypes.SequenceEqual(candidateTypes)) return null;
            return cp;
        }
        catch
        {
            // Corrupt checkpoint → start over rather than crashing.
            return null;
        }
    }

    private static void SaveCheckpoint(
        string path, string sourceModelPath, string corpusName,
        IReadOnlyList<string> categories, IReadOnlyList<LlamaTensorType> candidateTypes,
        double baseline,
        Dictionary<(string Cat, LlamaTensorType Type), double> ablations)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var dict = ablations.ToDictionary(kv => $"{kv.Key.Cat}|{kv.Key.Type}", kv => kv.Value);
        var cp = new Checkpoint(sourceModelPath, corpusName, categories, candidateTypes,
            double.IsNaN(baseline) ? null : baseline, dict, DateTime.UtcNow);
        // Atomic write: stage to .tmp, then rename. Avoids a half-written
        // file if the process is killed mid-serialize.
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(cp, CheckpointJsonOpts));
        File.Move(tmp, path, overwrite: true);
    }

    // ---- helpers --------------------------------------------------------

    /// <summary>Produce an ablation recipe: target category → <paramref name="type"/>, everything else → F16.</summary>
    private static LlamaQuantRecipe BuildAblationRecipe(
        IReadOnlyList<string> tensors, string category, LlamaTensorType type)
    {
        var entries = new List<LlamaQuantRecipeEntry>(tensors.Count);
        foreach (var name in tensors)
        {
            var isTarget = CategoryMatch(name, category);
            entries.Add(new LlamaQuantRecipeEntry(
                TensorName:        name,
                ChosenType:        isTarget ? type : LlamaTensorType.F16,
                BitsPerElement:    isTarget ? GetBitsPerElement(type) : 16.0,
                RelativeMse:       0.0,
                ExceededThreshold: false));
        }
        return new LlamaQuantRecipe(
            Threshold: 0.0,
            SourceScoreTablePath: null,
            Entries: entries,
            BuiltAtUtc: DateTime.UtcNow);
    }

    /// <summary>
    /// Tensor-name category matcher. Mirrors the categorization used in
    /// the Run 9/11 investigation scripts and llama-quant.cpp's
    /// <c>tensor_get_category</c>.
    /// </summary>
    private static bool CategoryMatch(string tensorName, string category) =>
        category.Contains('.')
            ? tensorName.EndsWith(category, StringComparison.Ordinal)
            : tensorName.Contains(category, StringComparison.Ordinal);

    private static double GetBitsPerElement(LlamaTensorType t) => t switch
    {
        LlamaTensorType.F32     => 32.0,
        LlamaTensorType.F16     => 16.0,
        LlamaTensorType.BF16    => 16.0,
        LlamaTensorType.Q8_0    => 8.5,
        LlamaTensorType.Q6_K    => 6.5625,
        LlamaTensorType.Q5_K    => 5.5,
        LlamaTensorType.Q4_K    => 4.5,
        LlamaTensorType.IQ4_XS  => 4.25,
        LlamaTensorType.Q3_K    => 3.4375,
        LlamaTensorType.IQ3_S   => 3.4375,
        LlamaTensorType.Q2_K    => 2.625,
        LlamaTensorType.IQ2_S   => 2.5,
        _                       => 8.0,
    };

    /// <summary>Filesystem-safe slug for a category name like "attn_v.weight" → "attn_v_weight".</summary>
    private static string Slugify(string s) =>
        Regex.Replace(s, "[^A-Za-z0-9_-]", "_");

    private static string ResolveArchitecture(LlamaGgufFile file)
    {
        var entry = file.Metadata.FirstOrDefault(m => m.Key == "general.architecture");
        return entry is not null && entry.Value.Type == LlamaGgufType.String
            ? entry.Value.AsString()
            : "unknown";
    }

    private static int ResolveLayerCount(LlamaGgufFile file)
    {
        // Architecture-specific key — try the standard pattern first.
        var arch = ResolveArchitecture(file);
        var key = $"{arch}.block_count";
        var entry = file.Metadata.FirstOrDefault(m => m.Key == key);
        if (entry is null) return -1;
        return entry.Value.Type switch
        {
            LlamaGgufType.Uint32 => (int)entry.Value.AsUInt32(),
            LlamaGgufType.Int32  => entry.Value.AsInt32(),
            LlamaGgufType.Uint64 => (int)entry.Value.AsUInt64(),
            _                    => -1,
        };
    }

    /// <summary>
    /// Total parameter count = sum over all weight tensors of the
    /// product of their dimensions. Used by the recipe builder's
    /// size-scaling factor (target_params / source_params), so we
    /// store it in the profile's provenance rather than recomputing
    /// it later from a possibly-missing source file.
    /// </summary>
    private static long ResolveParameterCount(LlamaGgufFile file)
    {
        long total = 0;
        foreach (var t in file.Tensors)
        {
            long n = 1;
            foreach (var d in t.Dimensions) n *= (long)d;
            total += n;
        }
        return total;
    }

    private static readonly string BuilderVersionString =
        $"LlamaSensitivityProfileBuilder/{typeof(LlamaSensitivityProfileBuilder).Assembly.GetName().Version}";

    /// <summary>
    /// Quantize <paramref name="source"/> to <paramref name="output"/>
    /// with optional imatrix and optional recipe. Wrapper around
    /// <see cref="LlamaQuantizer.QuantizeAsync"/> that applies the
    /// recipe through <see cref="LlamaQuantRecipe.ToTtOverrides"/>.
    /// </summary>
    private static Task QuantizeAsync(
        string source, string output,
        LlamaFileType ftype,
        string? imatrixPath,
        LlamaQuantRecipe? recipe,
        CancellationToken ct)
    {
        var p = new LlamaQuantizationParameters
        {
            FileType        = ftype,
            ImatrixPath     = imatrixPath,
            AllowRequantize = true,
        };
        if (recipe is not null)
        {
            p.Pure = false;
            p.TensorTypeOverrides = recipe.ToTtOverrides();
        }
        return LlamaQuantizer.QuantizeAsync(source, output, p, ct);
    }
}
