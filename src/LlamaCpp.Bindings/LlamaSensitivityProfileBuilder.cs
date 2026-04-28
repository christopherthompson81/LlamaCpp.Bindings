using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings;

/// <summary>
/// Orchestrates per-category and per-layer PPL ablation campaigns.
/// Persistence is via <see cref="LlamaInvestigationDb"/>: every PPL
/// measurement (baseline + each ablation cell) lands as one row.
/// Resume across crashes/cancellations is automatic — re-running the
/// same campaign skips cells that already have a sample.
/// </summary>
/// <remarks>
/// <para>
/// Two campaign modes share the inner machinery (quantize → score →
/// record → cleanup loop):
/// </para>
/// <list type="bullet">
///   <item><see cref="BuildAsync"/> — per-category. Each ablation
///     pins a tensor category (e.g. <c>ffn_up</c>) to a candidate type
///     and everything else to F16. Returns a
///     <see cref="LlamaSensitivityProfile"/> snapshot suitable for
///     export. Cheap: 22 PPLs for the 3-type × 7-category default.</item>
///   <item><see cref="BuildPerLayerAsync"/> — per-tensor. Each
///     ablation pins one specific tensor (e.g.
///     <c>blk.13.attn_v.weight</c>) to a candidate type and everything
///     else to F16. Expensive: O(layers × categories × types) PPLs;
///     for a 28-layer model with 7 categories × 7 types that's ~1370
///     runs. Data lands in the DB; recipe builders query per-tensor
///     rows when present and fall back to per-category rows otherwise.</item>
/// </list>
/// </remarks>
public static class LlamaSensitivityProfileBuilder
{
    /// <summary>Tunable knobs shared by both campaign modes.</summary>
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
        /// Categories to score (per-category mode only). Default covers the
        /// 7 weight categories of a standard transformer.
        /// </summary>
        public IReadOnlyList<string> Categories { get; set; } = new[]
        {
            "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_output.weight", "ffn_up", "ffn_gate", "ffn_down",
        };

        /// <summary>Optional imatrix GGUF path to use for imatrix-aware quantization. Recommended.</summary>
        public string? ImatrixPath { get; set; }

        /// <summary>Concurrency cap for the inner PPL runner. <c>0</c> = auto.</summary>
        public int MaxConcurrent { get; set; } = 0;

        /// <summary>Override available GPU VRAM in bytes (default: 24 GB / RTX 3090 class).</summary>
        public long? AvailableVramBytes { get; set; }

        /// <summary>Working directory for temp quantized files; defaults to a fresh tempdir.</summary>
        public string? WorkingDirectory { get; set; }

        /// <summary>If true, delete the temp quants when the campaign finishes.</summary>
        public bool CleanupWorkingDirectory { get; set; } = true;

        /// <summary>PPL options passed through to every inner perplexity run. Defaults to n_ctx=512.</summary>
        public LlamaPerplexityOptions? PerplexityOptions { get; set; }

        /// <summary>
        /// Per-category catastrophic threshold used to compute
        /// <see cref="LlamaSensitivityCategoryCoefficient.RecommendedFloor"/>.
        /// Per-category mode only.
        /// </summary>
        public double KneeDeltaPplThreshold { get; set; } = 5.0;

        /// <summary>
        /// Measurement database. <c>null</c> opens
        /// <see cref="LlamaInvestigationDb.DefaultPath"/> internally.
        /// Pass an explicit instance to share across builder calls
        /// (e.g. per-category followed by per-layer mode).
        /// </summary>
        public LlamaInvestigationDb? MeasurementDb { get; set; }

        /// <summary>Optional GPU model string recorded per row (e.g. "RTX 3090").</summary>
        public string? GpuModel { get; set; }
    }

    /// <summary>
    /// Progress event raised as the campaign advances. Carries both
    /// global-counter info (for an overall progress bar) and optional
    /// cell-level info (for the per-cell progress grid in the UI).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Cell-level fields are populated whenever an individual ablation
    /// cell transitions state (Quantizing → Scoring → Done). Listeners
    /// that don't care about per-cell granularity can ignore them and
    /// just track <see cref="CompletedJobs"/> / <see cref="TotalJobs"/>.
    /// </para>
    /// <para>
    /// The "Plan" event is fired exactly once at the start of each
    /// campaign carrying the full target list — UIs use it to allocate
    /// the progress grid rows once and avoid reshapes thereafter.
    /// </para>
    /// </remarks>
    public sealed record Progress(
        Stage Stage,
        int CompletedJobs,
        int TotalJobs,
        string? CurrentLabel = null,
        /// <summary>Ablation target (e.g. "category:ffn_up", "tensor:blk.13.attn_v.weight"). Null for non-cell events.</summary>
        string? CellTarget = null,
        /// <summary>Candidate type for the cell. Null for non-cell events.</summary>
        LlamaTensorType? CellType = null,
        /// <summary>Per-cell state transition. Null for non-cell events.</summary>
        CellState? CellState = null,
        /// <summary>Measured ΔPPL when <see cref="CellState"/> is Done.</summary>
        double? CellDelta = null,
        /// <summary>For <see cref="Stage.Plan"/>: the complete list of (target, type) cells the campaign will run, in deterministic order.</summary>
        IReadOnlyList<(string Target, LlamaTensorType Type)>? Plan = null,
        /// <summary>
        /// Monotonic progress fraction in <c>[0, 1]</c>, treating each
        /// cell as contributing two work units (one for quantize, one
        /// for score). Always advances; never rewinds when the campaign
        /// transitions from a quantize batch to its score batch. UIs
        /// should bind progress bars to this rather than computing
        /// <c>CompletedJobs / TotalJobs</c>, since those count post-PPL
        /// cells and don't move during the quantize phase.
        /// </summary>
        double Fraction = 0.0);

    /// <summary>Coarse phases of the campaign for progress reporting.</summary>
    public enum Stage
    {
        /// <summary>Initial event listing every cell the campaign will run. Fired once.</summary>
        Plan,
        Quantizing,
        Scoring,
        Done,
    }

    /// <summary>
    /// Per-cell lifecycle state. Drives the progress-grid cell glyphs in
    /// the UI: <c>Pending</c> → <c>Quantizing</c> → <c>Scoring</c> →
    /// <c>Done</c> (with delta), or <c>Resumed</c> when a cell was
    /// already in the DB at campaign start.
    /// </summary>
    public enum CellState
    {
        Pending,
        Resumed,
        Quantizing,
        Scoring,
        Done,
        Errored,
    }

    /// <summary>Sentinel <see cref="LlamaMeasurementRecord.AblationTarget"/> for the F16 baseline measurement.</summary>
    public const string BaselineTarget = "baseline";

    // ---------------------------------------------------------------- //
    // Public entry points                                              //
    // ---------------------------------------------------------------- //

    /// <summary>Build a sensitivity profile for <paramref name="sourceModelPath"/> by per-category ablation.</summary>
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
        var ggufFile = LlamaGgufFile.Open(sourceModelPath);
        var architectureId = ResolveArchitecture(ggufFile);
        var layerCount = ResolveLayerCount(ggufFile);
        var paramCount = ResolveParameterCount(ggufFile);
        var weightTensors = WeightTensorNames(ggufFile);

        // Validate every requested category has at least one tensor.
        foreach (var c in opts.Categories)
        {
            if (!weightTensors.Any(n => CategoryMatch(n, c)))
            {
                throw new InvalidOperationException(
                    $"Category '{c}' has no matching tensors in {sourceModelPath} — typo, or this " +
                    "architecture doesn't have that category. Adjust Options.Categories.");
            }
        }

        // Build per-category specs: target = "category:<name>", isAblated
        // matches the category's tensor name pattern.
        var specs = new List<AblationSpec>();
        foreach (var cat in opts.Categories)
        {
            var capturedCat = cat;
            foreach (var type in opts.CandidateTypes)
                specs.Add(new AblationSpec(
                    Target:    $"category:{capturedCat}",
                    Type:      type,
                    IsAblated: name => CategoryMatch(name, capturedCat)));
        }

        var results = await ExecuteCampaignAsync(
            sourceModelPath, corpusPath, opts, weightTensors,
            architectureId, paramCount, specs, progress, cancellationToken)
            .ConfigureAwait(false);

        // Build the per-category coefficient records from results.
        var pplOpts = opts.PerplexityOptions ?? new LlamaPerplexityOptions { ContextSize = 512 };
        var categories = new Dictionary<string, LlamaSensitivityCategoryCoefficient>();
        foreach (var cat in opts.Categories)
        {
            var deltas = new Dictionary<LlamaTensorType, double>();
            foreach (var type in opts.CandidateTypes)
            {
                if (results.AblationPpl.TryGetValue(($"category:{cat}", type), out var ppl))
                    deltas[type] = ppl - results.Baseline;
            }

            LlamaTensorType? floor = null;
            foreach (var type in opts.CandidateTypes.OrderBy(t => GetBitsPerElement(t)))
            {
                if (deltas.TryGetValue(type, out var d) && d <= opts.KneeDeltaPplThreshold)
                {
                    floor = type;
                    break;
                }
            }
            categories[cat] = new LlamaSensitivityCategoryCoefficient(deltas, floor);
        }

        var provenance = new LlamaSensitivityProvenance(
            Method:               "ablation",
            SourceModel:          Path.GetFileName(sourceModelPath),
            SourceParameterCount: paramCount,
            Corpus:               Path.GetFileName(corpusPath),
            BuiltAtUtc:           DateTime.UtcNow,
            BuilderVersion:       BuilderVersionString);
        return new LlamaSensitivityProfile(
            SchemaVersion:         LlamaSensitivityProfile.CurrentSchemaVersion,
            ArchitectureId:        architectureId,
            LayerCount:            layerCount,
            FamilyNotes:           null,
            Provenance:            provenance,
            F16BaselinePerplexity: results.Baseline,
            BaselineContextSize:   pplOpts.ContextSize,
            Categories:            categories);
    }

    /// <summary>
    /// Run a per-tensor ablation campaign. Each spec ablates one
    /// individual tensor (not a whole category) at a candidate type;
    /// data accumulates in <see cref="LlamaInvestigationDb"/> for later
    /// recipe construction. Returns the count of measurements recorded
    /// during this call (not including baseline or skipped/resumed cells).
    /// </summary>
    /// <param name="targetTensors">
    /// Tensor names to ablate. <c>null</c> auto-derives from
    /// <see cref="LlamaArchitectureRegistry"/> for the model's
    /// architecture, including all per-layer tensors and top-level
    /// quantizable tensors. Tensors not present in the source model
    /// are silently filtered out.
    /// </param>
    public static async Task<int> BuildPerLayerAsync(
        string sourceModelPath,
        string corpusPath,
        IReadOnlyList<string>? targetTensors = null,
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
        var ggufFile = LlamaGgufFile.Open(sourceModelPath);
        var architectureId = ResolveArchitecture(ggufFile);
        var layerCount = ResolveLayerCount(ggufFile);
        var paramCount = ResolveParameterCount(ggufFile);
        var weightTensors = WeightTensorNames(ggufFile);

        // Auto-derive target tensors from the architecture registry when
        // none provided. Includes both per-layer (blk.{i}.*) and top-level
        // (output.weight, token_embd.weight) quantizable tensors.
        if (targetTensors is null || targetTensors.Count == 0)
        {
            var spec = LlamaArchitectureRegistry.Lookup(architectureId)
                    ?? LlamaArchitectureRegistry.StandardTransformer;
            var layers = layerCount > 0 ? layerCount : 1;
            targetTensors = spec.ExpandPerLayerTensors(layers)
                .Concat(spec.TopLevelTensors)
                .ToList();
        }

        // Filter to tensors that actually exist in this model. Avoids
        // wasted PPL runs when the architecture registry includes
        // optional tensors (e.g. tied embeddings → no output.weight).
        var present = new HashSet<string>(weightTensors, StringComparer.Ordinal);
        var effective = targetTensors.Where(present.Contains).ToList();
        if (effective.Count == 0)
        {
            throw new InvalidOperationException(
                "No matching tensors in source model. Either targetTensors is empty after filtering, " +
                "or the architecture registry returned templates that don't apply to this model.");
        }

        var specs = new List<AblationSpec>();
        foreach (var t in effective)
        {
            var capturedTensor = t;
            foreach (var type in opts.CandidateTypes)
                specs.Add(new AblationSpec(
                    Target:    $"tensor:{capturedTensor}",
                    Type:      type,
                    IsAblated: name => name == capturedTensor));
        }

        var results = await ExecuteCampaignAsync(
            sourceModelPath, corpusPath, opts, weightTensors,
            architectureId, paramCount, specs, progress, cancellationToken)
            .ConfigureAwait(false);
        return results.AblationPpl.Count;
    }

    // ---------------------------------------------------------------- //
    // Shared campaign runner                                           //
    // ---------------------------------------------------------------- //

    /// <summary>
    /// One ablation cell: which tensors get pinned to which candidate
    /// type. The campaign runner builds a recipe from this by sending
    /// every tensor through <see cref="IsAblated"/> — true → pinned to
    /// <see cref="Type"/>, false → F16.
    /// </summary>
    private sealed record AblationSpec(
        string Target,
        LlamaTensorType Type,
        Predicate<string> IsAblated);

    /// <summary>Aggregated results from a campaign — the in-memory mirror of what just got written to the DB.</summary>
    private sealed record CampaignResults(
        double Baseline,
        Dictionary<(string Target, LlamaTensorType Type), double> AblationPpl);

    /// <summary>
    /// Run a sequence of ablation specs against <paramref name="sourceModelPath"/>.
    /// Handles identity computation, baseline measurement, batched
    /// quantize+PPL pipeline, DB persistence, and resume.
    /// </summary>
    private static async Task<CampaignResults> ExecuteCampaignAsync(
        string sourceModelPath,
        string corpusPath,
        Options opts,
        IReadOnlyList<string> weightTensors,
        string architectureId,
        long paramCount,
        IReadOnlyList<AblationSpec> specs,
        IProgress<Progress>? progress,
        CancellationToken cancellationToken)
    {
        var corpusText = await File.ReadAllTextAsync(corpusPath, cancellationToken).ConfigureAwait(false);
        var workDir = opts.WorkingDirectory ?? Path.Combine(
            Path.GetTempPath(),
            "llama-sensitivity-profile-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workDir);

        var ownsDb = opts.MeasurementDb is null;
        var db = opts.MeasurementDb ?? LlamaInvestigationDb.Open();

        try
        {
            var modelSha   = LlamaInvestigationDb.ComputeContentSha(sourceModelPath);
            var corpusSha  = LlamaInvestigationDb.ComputeTextSha(corpusText);
            var imatrixSha = string.IsNullOrEmpty(opts.ImatrixPath)
                ? LlamaInvestigationDb.NoImatrixSha
                : LlamaInvestigationDb.ComputeContentSha(opts.ImatrixPath);
            var corpusName = Path.GetFileName(corpusPath);
            var pplOpts = opts.PerplexityOptions ?? new LlamaPerplexityOptions { ContextSize = 512 };

            int batchSize = opts.MaxConcurrent > 0
                ? opts.MaxConcurrent
                : Math.Min(Environment.ProcessorCount, 8);

            int totalJobs = 1 + specs.Count;
            int completed = 0;
            int scored = 0;    // assigned again when baseline path is resolved

            // Monotonic progress fraction. Each cell contributes two
            // work units (quantize + score), so the bar advances
            // smoothly through both phases instead of rewinding when a
            // batch transitions from quantize to score.
            // Clamp to [0, 1] — resumed-cells bookkeeping can briefly
            // make the numerator overshoot before completed catches up
            // post-resume-loop, which we'd rather hide than expose.
            double Fraction()
            {
                if (totalJobs <= 0) return 0.0;
                var f = (completed + scored) / (2.0 * totalJobs);
                return f < 0.0 ? 0.0 : f > 1.0 ? 1.0 : f;
            }

            // Plan event: tells the UI exactly which (target, type)
            // cells to allocate in the progress grid. Fired once,
            // before any work — UI uses this to pre-build the grid
            // shape so cell updates don't reshape the layout.
            var planCells = specs.Select(s => (s.Target, s.Type)).ToList();
            progress?.Report(new Progress(
                Stage.Plan, 0, totalJobs,
                CurrentLabel: $"{planCells.Count} ablation cells planned",
                Plan: planCells,
                Fraction: 0.0));

            void ReportQuant(string label) =>
                progress?.Report(new Progress(
                    Stage.Quantizing, completed, totalJobs, label,
                    Fraction: Fraction()));

            void ReportCell(AblationSpec spec, CellState state, double? delta = null) =>
                progress?.Report(new Progress(
                    state == CellState.Quantizing ? Stage.Quantizing : Stage.Scoring,
                    completed, totalJobs,
                    CurrentLabel: $"{spec.Target} @ {spec.Type}",
                    CellTarget: spec.Target,
                    CellType:   spec.Type,
                    CellState:  state,
                    CellDelta:  delta,
                    Fraction:   Fraction()));

            // ---- Resume from DB ----
            // Pull every existing measurement that matches this campaign
            // signature (model+corpus+imatrix+ctx). One row per
            // (target, type); duplicates accumulated by re-runs are
            // ignored at the cache level (TryAdd) — the DB still has them.
            var ablationPpl = new Dictionary<(string Target, LlamaTensorType Type), double>();
            double? resumedBaseline = null;
            var targetsInThisCampaign = new HashSet<string>(specs.Select(s => s.Target), StringComparer.Ordinal);
            foreach (var existing in db.Query(new LlamaMeasurementFilter
            {
                ModelSha = modelSha, CorpusSha = corpusSha, ImatrixSha = imatrixSha,
                ContextSize = pplOpts.ContextSize,
            }))
            {
                if (existing.AblationTarget == BaselineTarget)
                {
                    resumedBaseline ??= existing.AblationPpl;
                    continue;
                }
                if (!targetsInThisCampaign.Contains(existing.AblationTarget)) continue;
                if (!opts.CandidateTypes.Contains(existing.AblationType)) continue;
                ablationPpl.TryAdd((existing.AblationTarget, existing.AblationType), existing.AblationPpl);
            }

            // ---- Baseline ----
            double baseline;
            if (resumedBaseline is double cachedBaseline)
            {
                baseline = cachedBaseline;
                completed++;
                scored = 1 + ablationPpl.Count;
                progress?.Report(new Progress(Stage.Scoring, scored, totalJobs,
                    CurrentLabel: $"resumed from DB (baseline + {ablationPpl.Count} ablations)",
                    Fraction: Fraction()));
            }
            else
            {
                var baselinePath = Path.Combine(workDir, "baseline.gguf");
                ReportQuant("baseline (F16)");
                await QuantizeAsync(sourceModelPath, baselinePath,
                    ftype: LlamaFileType.MostlyF16,
                    imatrixPath: opts.ImatrixPath,
                    recipe: null,
                    cancellationToken).ConfigureAwait(false);
                completed++;

                progress?.Report(new Progress(Stage.Scoring, 0, totalJobs,
                    CurrentLabel: $"baseline (batch={batchSize}, ppl-concurrency=auto)",
                    Fraction: Fraction()));
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
                progress?.Report(new Progress(Stage.Scoring, scored, totalJobs,
                    CurrentLabel: "baseline done",
                    Fraction: Fraction()));
            }

            // ---- Ablation specs, batched ----
            // Bump completed BEFORE emitting Resumed events so the
            // monotonic Fraction() includes them — otherwise the bar
            // briefly understates progress at startup before the
            // batch loop catches up.
            completed += ablationPpl.Count;

            foreach (var s in specs)
            {
                if (ablationPpl.TryGetValue((s.Target, s.Type), out var ppl))
                    progress?.Report(new Progress(
                        Stage.Scoring, completed, totalJobs,
                        CurrentLabel: $"{s.Target} @ {s.Type} (resumed)",
                        CellTarget:   s.Target,
                        CellType:     s.Type,
                        CellState:    CellState.Resumed,
                        CellDelta:    ppl - baseline,
                        Fraction:     Fraction()));
            }

            var pendingSpecs = specs
                .Where(s => !ablationPpl.ContainsKey((s.Target, s.Type)))
                .ToList();

            for (int batchStart = 0; batchStart < pendingSpecs.Count; batchStart += batchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var batch = pendingSpecs.Skip(batchStart).Take(batchSize).ToList();
                var batchPaths = new List<(AblationSpec Spec, string Path)>();

                foreach (var s in batch)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var slug = $"{Slugify(s.Target)}_{s.Type}";
                    var outPath = Path.Combine(workDir, $"{slug}.gguf");
                    ReportCell(s, CellState.Quantizing);
                    var recipe = BuildAblationRecipe(weightTensors, s);
                    await QuantizeAsync(sourceModelPath, outPath,
                        ftype: LlamaFileType.Q4_K_M,
                        imatrixPath: opts.ImatrixPath,
                        recipe: recipe,
                        cancellationToken).ConfigureAwait(false);
                    batchPaths.Add((s, outPath));
                    ReportCell(s, CellState.Scoring);
                    completed++;
                }

                var batchJobs = batchPaths.Select((bp, idx) =>
                    new LlamaPerplexity.PerplexityJob(
                        ModelPath: bp.Path, Corpus: corpusText, Options: pplOpts,
                        // Tag carries the batch index — splitting on '|'
                        // breaks for tensor:blk.0.attn_q.weight (target
                        // contains dots), so we look up by index instead.
                        Tag: idx.ToString())).ToList();

                await foreach (var jr in LlamaPerplexity.RunParallelAsync(
                    batchJobs, maxConcurrent: 0, cancellationToken: cancellationToken))
                {
                    scored++;
                    if (jr.Tag is string tag && int.TryParse(tag, out var idx))
                    {
                        var (spec, path) = batchPaths[idx];
                        var ablation = jr.Result.Perplexity;
                        ablationPpl[(spec.Target, spec.Type)] = ablation;
                        db.RecordMeasurement(BuildMeasurementRecord(
                            modelSha, architectureId, paramCount, corpusSha, corpusName,
                            imatrixSha, pplOpts.ContextSize,
                            target:        spec.Target,
                            ablationType:  spec.Type,
                            baselineType:  LlamaTensorType.F16,
                            baselinePpl:   baseline,
                            ablationPpl:   ablation,
                            deltaPpl:      ablation - baseline,
                            gpuModel:      opts.GpuModel));
                        progress?.Report(new Progress(
                            Stage.Scoring, scored, totalJobs,
                            CurrentLabel: $"{spec.Target} @ {spec.Type} = {ablation:F4}",
                            CellTarget:   spec.Target,
                            CellType:     spec.Type,
                            CellState:    CellState.Done,
                            CellDelta:    ablation - baseline,
                            Fraction:     Fraction()));
                        try { File.Delete(path); } catch { /* best-effort */ }
                    }
                }
            }

            progress?.Report(new Progress(Stage.Done, scored, totalJobs, Fraction: 1.0));
            return new CampaignResults(baseline, ablationPpl);
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

    /// <summary>
    /// Build an ablation recipe per the spec: every tensor that
    /// matches <see cref="AblationSpec.IsAblated"/> goes to
    /// <see cref="AblationSpec.Type"/>; everything else stays at F16.
    /// </summary>
    private static LlamaQuantRecipe BuildAblationRecipe(
        IReadOnlyList<string> tensors, AblationSpec spec)
    {
        var entries = new List<LlamaQuantRecipeEntry>(tensors.Count);
        foreach (var name in tensors)
        {
            var isTarget = spec.IsAblated(name);
            entries.Add(new LlamaQuantRecipeEntry(
                TensorName:        name,
                ChosenType:        isTarget ? spec.Type : LlamaTensorType.F16,
                BitsPerElement:    isTarget ? GetBitsPerElement(spec.Type) : 16.0,
                RelativeMse:       0.0,
                ExceededThreshold: false));
        }
        return new LlamaQuantRecipe(
            Threshold: 0.0,
            SourceScoreTablePath: null,
            Entries: entries,
            BuiltAtUtc: DateTime.UtcNow);
    }

    private static List<string> WeightTensorNames(LlamaGgufFile file) =>
        file.Tensors
            .Where(t => t.Dimensions.Length > 1 && t.Name.EndsWith(".weight", StringComparison.Ordinal))
            .Select(t => t.Name)
            .ToList();

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

    /// <summary>Filesystem-safe slug for a target name like "tensor:blk.0.attn_v.weight".</summary>
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
