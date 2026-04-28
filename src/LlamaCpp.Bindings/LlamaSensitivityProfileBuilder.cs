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
///         makes profile-building tractable.</item>
///   <item>Compute ΔPPL = ablation_ppl − baseline_ppl per cell.</item>
///   <item>Detect each category's "knee" — the smallest type before
///         ΔPPL crosses a catastrophic threshold (default 5.0 PPL).</item>
/// </list>
/// </summary>
/// <remarks>
/// <para>
/// Persistence: every PPL measurement (baseline + per-category) is
/// written to <see cref="LlamaInvestigationDb"/> as it lands. Resume
/// across crashes or cancellations is automatic — re-running the same
/// campaign skips cells that already have a sample in the DB. The DB
/// is the source of truth; the profile JSON returned from
/// <see cref="BuildAsync"/> is a derived snapshot.
/// </para>
/// </remarks>
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
        /// budget to pick a value that scales with model size.
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
        /// Per-category catastrophic threshold used to compute
        /// <see cref="LlamaSensitivityCategoryCoefficient.RecommendedFloor"/>.
        /// A type is below the floor if its ΔPPL exceeds this value;
        /// recipes won't choose below the floor. 5.0 is a generous
        /// default — picks up genuine knees (Run 11 saw 124 PPL on
        /// `ffn_down` at Q2_K) without flagging mild degradation as a
        /// knee.
        /// </summary>
        public double KneeDeltaPplThreshold { get; set; } = 5.0;

        /// <summary>
        /// Measurement database. Every PPL result lands here as one row;
        /// resume across crashes/cancellations skips cells that already
        /// have a sample. <c>null</c> (default) opens
        /// <see cref="LlamaInvestigationDb.DefaultPath"/> internally and
        /// disposes it when the campaign finishes. Pass an explicit
        /// instance to share a DB across multiple builder calls (e.g.
        /// per-category followed by per-layer mode).
        /// </summary>
        public LlamaInvestigationDb? MeasurementDb { get; set; }

        /// <summary>
        /// Optional GPU model string recorded in each measurement row
        /// (e.g. "RTX 3090"). Helps disambiguate cross-environment
        /// comparisons over time. <c>null</c> leaves the field unset.
        /// </summary>
        public string? GpuModel { get; set; }
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

    /// <summary>Sentinel <see cref="LlamaMeasurementRecord.AblationTarget"/> for the F16 baseline measurement.</summary>
    public const string BaselineTarget = "baseline";

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

        // Open or borrow the measurement DB. If we open it ourselves we
        // own its lifetime; if the caller passed one we leave them to
        // dispose it.
        var ownsDb = opts.MeasurementDb is null;
        var db = opts.MeasurementDb ?? LlamaInvestigationDb.Open();

        try
        {
            // --- 1. Compute content-stable identity for this campaign.
            //     These hashes key every row; renaming/relocating the
            //     model file later still matches its measurements.
            var modelSha   = LlamaInvestigationDb.ComputeContentSha(sourceModelPath);
            var corpusSha  = LlamaInvestigationDb.ComputeTextSha(corpusText);
            var imatrixSha = string.IsNullOrEmpty(opts.ImatrixPath)
                ? LlamaInvestigationDb.NoImatrixSha
                : LlamaInvestigationDb.ComputeContentSha(opts.ImatrixPath);
            var corpusName = Path.GetFileName(corpusPath);

            // --- 2. Enumerate the source's 2-D weight tensors and group
            //        them by category so we can build per-category recipes.
            var ggufFile = LlamaGgufFile.Open(sourceModelPath);
            var architectureId = ResolveArchitecture(ggufFile);
            var layerCount = ResolveLayerCount(ggufFile);
            var paramCount = ResolveParameterCount(ggufFile);

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

            // --- 3. Build the F16 baseline + every (category, type) ablation
            //        quant. All written into workDir as <label>.gguf.
            var baselinePath = Path.Combine(workDir, "baseline.gguf");
            int totalJobs = 1 + opts.Categories.Count * opts.CandidateTypes.Count;
            int completed = 0;

            void ReportQuant(string label) =>
                progress?.Report(new Progress(Stage.Quantizing, completed, totalJobs, label));

            int batchSize = opts.MaxConcurrent > 0
                ? opts.MaxConcurrent
                : Math.Min(Environment.ProcessorCount, 8);

            // Default to n_ctx=512 to match the wikitext-2 published-number
            // convention used in Run 9/11 (and what GGUFLab's standalone
            // Perplexity tool uses).
            var pplOpts = opts.PerplexityOptions ?? new LlamaPerplexityOptions { ContextSize = 512 };

            // ---- Resumability via DB ----
            // Pull every existing measurement for this exact campaign
            // signature into an in-memory dict so the inner loop's
            // skip-check is O(1). The DB is the source of truth across
            // process restarts; the dict is just a cache for this run.
            var ablationPpl = new Dictionary<(string Cat, LlamaTensorType Type), double>();
            double? resumedBaseline = null;
            foreach (var existing in db.Query(new LlamaMeasurementFilter
            {
                ModelSha = modelSha, CorpusSha = corpusSha, ImatrixSha = imatrixSha,
                ContextSize = pplOpts.ContextSize,
            }))
            {
                if (existing.AblationTarget == BaselineTarget)
                {
                    // Latest baseline wins (Query returns DESC by date).
                    resumedBaseline ??= existing.AblationPpl;
                    continue;
                }
                if (!existing.AblationTarget.StartsWith("category:", StringComparison.Ordinal))
                    continue;    // tensor: rows belong to per-layer mode
                var catName = existing.AblationTarget["category:".Length..];
                if (!opts.Categories.Contains(catName)) continue;
                if (!opts.CandidateTypes.Contains(existing.AblationType)) continue;
                ablationPpl.TryAdd((catName, existing.AblationType), existing.AblationPpl);
            }

            // First: baseline. One file in flight, then deleted. If the
            // DB already has it, skip both quantize and PPL.
            double baseline;
            int scored;
            if (resumedBaseline is double cachedBaseline)
            {
                baseline = cachedBaseline;
                completed++;
                scored = 1 + ablationPpl.Count;
                progress?.Report(new Progress(Stage.Scoring, scored, totalJobs,
                    CurrentLabel: $"resumed from DB (baseline + {ablationPpl.Count} ablations)"));
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
                    CurrentLabel: $"baseline (batch={batchSize}, ppl-concurrency=auto)"));
                var baselineJob = new LlamaPerplexity.PerplexityJob(
                    ModelPath: baselinePath, Corpus: corpusText, Options: pplOpts, Tag: "BASELINE");
                baseline = double.NaN;
                await foreach (var jr in LlamaPerplexity.RunParallelAsync(
                    new[] { baselineJob }, maxConcurrent: 1, cancellationToken: cancellationToken))
                {
                    baseline = jr.Result.Perplexity;
                }
                try { File.Delete(baselinePath); } catch { /* best-effort */ }
                if (double.IsNaN(baseline))
                    throw new InvalidOperationException("Baseline PPL never returned from runner.");

                db.RecordMeasurement(BuildMeasurementRecord(
                    modelSha, architectureId, paramCount, corpusSha, corpusName,
                    imatrixSha, pplOpts.ContextSize,
                    target:        BaselineTarget,
                    ablationType:  LlamaTensorType.F16,
                    baselineType:  LlamaTensorType.F16,
                    baselinePpl:   baseline,
                    ablationPpl:   baseline,
                    deltaPpl:      0.0,
                    gpuModel:      opts.GpuModel));
                scored = 1;
                progress?.Report(new Progress(Stage.Scoring, scored, totalJobs, CurrentLabel: "baseline done"));
            }

            // Now the ablation specs, batched into rounds of batchSize.
            // Skip any (cat, type) the DB already has.
            var allSpecs = new List<(string Category, LlamaTensorType Type)>();
            foreach (var category in opts.Categories)
                foreach (var type in opts.CandidateTypes)
                    if (!ablationPpl.ContainsKey((category, type)))
                        allSpecs.Add((category, type));
            completed += ablationPpl.Count;

            for (int batchStart = 0; batchStart < allSpecs.Count; batchStart += batchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var batch = allSpecs.Skip(batchStart).Take(batchSize).ToList();
                var batchPaths = new List<(string Cat, LlamaTensorType Type, string Path)>();

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

                var batchJobs = batchPaths.Select(bp =>
                    new LlamaPerplexity.PerplexityJob(
                        ModelPath: bp.Path, Corpus: corpusText, Options: pplOpts,
                        Tag: $"{bp.Cat}|{bp.Type}|{bp.Path}")).ToList();

                await foreach (var jr in LlamaPerplexity.RunParallelAsync(
                    batchJobs, maxConcurrent: 0, cancellationToken: cancellationToken))
                {
                    scored++;
                    if (jr.Tag is string tag)
                    {
                        var split = tag.Split('|');
                        var cat = split[0];
                        var type = Enum.Parse<LlamaTensorType>(split[1]);
                        var path = split[2];
                        var ablation = jr.Result.Perplexity;
                        ablationPpl[(cat, type)] = ablation;
                        db.RecordMeasurement(BuildMeasurementRecord(
                            modelSha, architectureId, paramCount, corpusSha, corpusName,
                            imatrixSha, pplOpts.ContextSize,
                            target:        $"category:{cat}",
                            ablationType:  type,
                            baselineType:  LlamaTensorType.F16,
                            baselinePpl:   baseline,
                            ablationPpl:   ablation,
                            deltaPpl:      ablation - baseline,
                            gpuModel:      opts.GpuModel));
                        progress?.Report(new Progress(Stage.Scoring, scored, totalJobs,
                            CurrentLabel: $"{cat} @ {type} = {ablation:F4}"));
                        try { File.Delete(path); } catch { /* best-effort */ }
                    }
                }
            }

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
                SourceParameterCount: paramCount,
                Corpus:               corpusName,
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
            if (ownsDb) db.Dispose();
            if (opts.CleanupWorkingDirectory)
            {
                try { Directory.Delete(workDir, recursive: true); }
                catch { /* best-effort */ }
            }
        }
    }

    private static LlamaMeasurementRecord BuildMeasurementRecord(
        string modelSha, string archId, long paramCount,
        string corpusSha, string corpusName, string imatrixSha, int contextSize,
        string target, LlamaTensorType ablationType, LlamaTensorType baselineType,
        double baselinePpl, double ablationPpl, double deltaPpl,
        string? gpuModel) =>
        new(
            ModelSha:        modelSha,
            ArchId:          archId,
            ParamCount:      paramCount,
            CorpusSha:       corpusSha,
            CorpusName:      corpusName,
            ImatrixSha:      imatrixSha,
            ContextSize:     contextSize,
            AblationTarget:  target,
            AblationType:    ablationType,
            BaselineType:    baselineType,
            BaselinePpl:     baselinePpl,
            AblationPpl:     ablationPpl,
            DeltaPpl:        deltaPpl,
            MeasuredAtUtc:   DateTime.UtcNow,
            BuilderVersion:  BuilderVersionString,
            LlamaCppVersion: LlamaCppVersionInfo.GitDescribe,
            GpuModel:        gpuModel,
            Notes:           null);

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
            ? tensorName == category ||
              tensorName.EndsWith("." + category, StringComparison.Ordinal)
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
    /// product of their dimensions.
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

    /// <summary>Quantize <paramref name="source"/> to <paramref name="output"/> with optional imatrix and recipe.</summary>
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
