using System.Text.RegularExpressions;
using System.Threading.Channels;

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

        /// <summary>
        /// Skip the per-cell quantize+load+free cycle and apply each
        /// ablation as a tensor-data swap on a persistent F16 model
        /// (<see cref="LlamaInPlaceAblator"/>). Each consumer worker
        /// loads the F16 model once, then rewrites only the ablated
        /// tensors' bytes between PPL passes.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Default <c>true</c>. Saves the disk space the disk path
        /// otherwise consumes for per-cell GGUFs (peak ~5–7 ablation
        /// files × source size in flight) and matches the disk path's
        /// wall time within ~3 % on per-category sweeps; on per-tensor
        /// sweeps (1 tensor per cell) it pulls ahead by skipping the
        /// 22 s per-cell load+free overhead. Set to <c>false</c> to
        /// fall back to the disk path when needed (e.g. when running
        /// against a llama.cpp pin whose internals haven't been
        /// re-validated against the shim — see Run 24 for details).
        /// </para>
        /// <para>
        /// PPL agrees with the disk path within ~0.002 PPL (verified
        /// on Qwen3-0.6B / Q4_K and Q2_K). The small gap is from
        /// F16-kernel vs Q-quant-kernel matmul choice on data that is
        /// bit-identical post-dequant. Recipe-vs-recipe deltas — the
        /// surface AQ decisions are made on — are unaffected since
        /// both legs use identical kernels.
        /// </para>
        /// <para>
        /// Requires the <c>libllamashim</c> native binary alongside
        /// the fetched libllama.so (build via
        /// <c>tools/native-shims/build.sh</c>). Failure to provide the
        /// shim surfaces as a <see cref="DllNotFoundException"/> on the
        /// first ablation.
        /// </para>
        /// </remarks>
        public bool UseInPlaceAblator { get; set; } = true;

        /// <summary>
        /// When <see cref="UseInPlaceAblator"/> is on, route ablations
        /// targeting the lm-head tensor through the disk-quantize path
        /// instead of the in-place path. Default <c>true</c>.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The in-place path encodes the round-trip Q-quant values back
        /// as F16 and runs the F16-kernel matmul at inference. For
        /// weight tensors deep in the network this agrees with the
        /// Q-kernel matmul to F32 reduction-order noise (~0.002 PPL,
        /// Run 24). For the lm-head matmul — whose output feeds the
        /// final softmax — the kernel choice has a much larger effect:
        /// in-place under-predicts the cost of low-bpw quantization on
        /// <c>token_embd.weight</c> (under tied embeddings) by up to
        /// 0.55 PPL at Q2_K (Run 27 / issue #48).
        /// </para>
        /// <para>
        /// "lm-head tensor" means <c>output.weight</c> always, plus
        /// <c>token_embd.weight</c> when the architecture has tied
        /// embeddings (no separate <c>output.weight</c> in the model).
        /// In per-category mode that's at most one cell per candidate
        /// type; in per-tensor mode it's the same single tensor across
        /// the candidate types. Disk-path cost on those few cells is
        /// the usual ~22 s vs ~10 s — small total impact, large
        /// correctness gain at low-bpw budgets.
        /// </para>
        /// <para>
        /// Set to <c>false</c> for pure-speed campaigns where the user
        /// has confirmed they don't care about low-bpw lm-head
        /// calibration accuracy (e.g., they're targeting Q4_K_M-class
        /// budgets where the optimizer pins lm-head at Q4_K via the
        /// stock baseline floor anyway).
        /// </para>
        /// </remarks>
        public bool DiskFallbackForLmHeadAblations { get; set; } = true;
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

            // PPL-side concurrency. In the old batch design this was also
            // the quantize batch size; in the continuous-flow design
            // quantize is always serial (CPU-saturating internally) and
            // this is just the consumer-pool size.
            int pplConcurrency = opts.MaxConcurrent > 0
                ? opts.MaxConcurrent
                : LlamaPerplexity.RecommendConcurrency(
                    new[] { sourceModelPath },
                    availableVramBytes: opts.AvailableVramBytes);

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
                    CurrentLabel: $"baseline (ppl-concurrency={pplConcurrency})",
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

            // Each consumer task uses pplOpts directly. The thread budget
            // matches what RunParallelAsync would have computed:
            // ProcessorCount / pplConcurrency. We snapshot it here once
            // so all consumers share the same setting.
            int autoThreadsPerJob = Math.Max(1, Environment.ProcessorCount / Math.Max(1, pplConcurrency));
            var consumerPplOpts = pplOpts.ThreadCount > 0
                ? pplOpts
                : new LlamaPerplexityOptions
                {
                    ContextSize            = pplOpts.ContextSize,
                    ScoreSecondHalfOnly    = pplOpts.ScoreSecondHalfOnly,
                    AddBeginningOfSequence = pplOpts.AddBeginningOfSequence,
                    ThreadCount            = autoThreadsPerJob,
                };

            // All shared-state mutations (counter increments, dict +
            // DB writes, progress reports) go through this lock. The
            // Report callback may post to a SynchronizationContext but
            // is itself fast, so holding the lock briefly is fine.
            var sharedLock = new object();

            // Branch on the campaign mode. UseInPlaceAblator skips the
            // per-cell GGUF quantize and per-cell GPU model load,
            // applying each ablation as a tensor-data swap on a
            // persistent F16 model — see LlamaInPlaceAblator.
            if (opts.UseInPlaceAblator)
            {
                await RunInPlaceCampaignAsync(
                    sourceModelPath, corpusText, opts, weightTensors,
                    pendingSpecs, baseline, modelSha, architectureId, paramCount,
                    corpusSha, corpusName, imatrixSha, pplOpts, consumerPplOpts,
                    pplConcurrency, db, ablationPpl, sharedLock,
                    workDir,
                    () => Fraction(),
                    setScored: v => { scored = v; },
                    getScored: () => scored,
                    totalJobs, progress,
                    cancellationToken).ConfigureAwait(false);

                progress?.Report(new Progress(Stage.Done, scored, totalJobs, Fraction: 1.0));
                return new CampaignResults(baseline, ablationPpl);
            }

            // Continuous-flow pipeline. Quantize is serial (one writer
            // saturating CPU); PPL runs at pplConcurrency in parallel.
            // A bounded channel hands quantized files off to consumers
            // and back-pressures the producer when all consumers are
            // busy, so peak files-on-disk stays bounded.
            //
            // Peak in flight = 1 (producer mid-write)
            //                + channelCapacity (queued, ready for PPL)
            //                + pplConcurrency  (consumers actively scoring).
            // We size channelCapacity so the total = pplConcurrency + 1,
            // matching the old design's "1 baseline + batchSize" peak.
            int channelCapacity = 1;
            var channel = Channel.CreateBounded<(AblationSpec Spec, string Path)>(
                new BoundedChannelOptions(channelCapacity)
                {
                    FullMode = BoundedChannelFullMode.Wait,
                    SingleReader = false,
                    SingleWriter = true,
                });

            var producerTask = Task.Run(async () =>
            {
                try
                {
                    foreach (var s in pendingSpecs)
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
                        lock (sharedLock) { completed++; }
                        ReportCell(s, CellState.Scoring);
                        await channel.Writer.WriteAsync((s, outPath), cancellationToken).ConfigureAwait(false);
                    }
                }
                finally
                {
                    channel.Writer.TryComplete();
                }
            }, cancellationToken);

            async Task ConsumerLoopAsync()
            {
                await foreach (var (spec, path) in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
                {
                    try
                    {
                        var modelOpts = new LlamaModelParameters { UseMmap = true, GpuLayerCount = -1 };
                        using var model = new LlamaModel(path, modelOpts);
                        var result = await LlamaPerplexity.ComputeAsync(
                            model, corpusText, consumerPplOpts,
                            progress: null, cancellationToken).ConfigureAwait(false);
                        var ablation = result.Perplexity;

                        lock (sharedLock)
                        {
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
                            scored++;
                            progress?.Report(new Progress(
                                Stage.Scoring, scored, totalJobs,
                                CurrentLabel: $"{spec.Target} @ {spec.Type} = {ablation:F4}",
                                CellTarget:   spec.Target,
                                CellType:     spec.Type,
                                CellState:    CellState.Done,
                                CellDelta:    ablation - baseline,
                                Fraction:     Fraction()));
                        }
                    }
                    finally
                    {
                        try { File.Delete(path); } catch { /* best-effort */ }
                    }
                }
            }

            var consumerTasks = Enumerable.Range(0, pplConcurrency)
                .Select(_ => Task.Run(ConsumerLoopAsync, cancellationToken))
                .ToArray();

            // Await producer first so any quantize/cancellation exception
            // surfaces here; consumers will then drain whatever's already
            // in the channel before exiting via the completion signal.
            try
            {
                await producerTask.ConfigureAwait(false);
            }
            catch
            {
                // Producer faulted (cancellation, quantize error, …).
                // Complete the channel so consumers wake up, then await
                // them swallowing any secondary failures — the producer
                // exception is the root cause we want to surface.
                channel.Writer.TryComplete();
                try { await Task.WhenAll(consumerTasks).ConfigureAwait(false); }
                catch { /* consumer faults are downstream of the producer fault */ }
                throw;
            }
            await Task.WhenAll(consumerTasks).ConfigureAwait(false);

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
    /// Materialize an ablation spec into the explicit list of
    /// <c>(tensorName, type)</c> pairs the in-place ablator consumes.
    /// Mirrors the recipe-building logic but without the F16-everywhere-else
    /// padding (the in-place ablator handles "everything else stays F16"
    /// implicitly via its source-data restoration step).
    /// </summary>
    private static List<(string TensorName, LlamaTensorType Type)>
        MaterializeAblationTensors(IReadOnlyList<string> tensors, AblationSpec spec)
    {
        var list = new List<(string, LlamaTensorType)>();
        foreach (var name in tensors)
        {
            if (spec.IsAblated(name)) list.Add((name, spec.Type));
        }
        return list;
    }

    /// <summary>
    /// Replacement producer/consumer loop for the in-place path. Each
    /// worker holds a persistent F16 LlamaModel + LlamaInPlaceAblator;
    /// the queue carries ablation specs (already-materialized into
    /// tensor-name lists) rather than file paths to fresh GGUFs.
    /// </summary>
    private static async Task RunInPlaceCampaignAsync(
        string sourceModelPath,
        string corpusText,
        Options opts,
        IReadOnlyList<string> weightTensors,
        IReadOnlyList<AblationSpec> pendingSpecs,
        double baseline,
        string modelSha,
        string architectureId,
        long paramCount,
        string corpusSha,
        string corpusName,
        string imatrixSha,
        LlamaPerplexityOptions pplOpts,
        LlamaPerplexityOptions consumerPplOpts,
        int pplConcurrency,
        LlamaInvestigationDb db,
        Dictionary<(string Target, LlamaTensorType Type), double> ablationPpl,
        object sharedLock,
        string workDir,
        Func<double> fraction,
        Action<int> setScored,
        Func<int> getScored,
        int totalJobs,
        IProgress<Progress>? progress,
        CancellationToken cancellationToken)
    {
        // Pre-load the imatrix once at campaign scope. All workers read
        // the same dictionary (by-tensor lookup) without contention,
        // since the dictionary itself is not mutated after construction.
        IReadOnlyDictionary<string, float[]>? sharedImatrix = null;
        if (!string.IsNullOrEmpty(opts.ImatrixPath))
        {
            sharedImatrix = LlamaCustomQuantizer.LoadImatrixForExternalUse(opts.ImatrixPath!);
        }

        // Materialize each pending spec into the explicit tensor list
        // up front, so workers don't replay the predicate on every
        // dequeue. Done once; cheap (O(specs × tensors)).
        var allItems = pendingSpecs
            .Select(s => (Spec: s, Tensors: MaterializeAblationTensors(weightTensors, s)))
            .ToList();

        // Issue #48: split items so any cell touching the lm-head goes
        // through the disk path. The in-place path's F16-encoded
        // round-trip + F16-kernel matmul under-predicts low-bpw
        // quantization cost on the lm-head matmul (output drives the
        // final softmax, where per-element rounding becomes per-logit
        // PPL drift). Disk path keeps the Q-storage and uses the
        // Q-kernel matmul, agreeing with what the user's deployed
        // model would actually compute.
        bool tiedEmbeddings = !weightTensors.Any(n => n == "output.weight");
        bool IsLmHeadCell(IReadOnlyList<(string Name, LlamaTensorType Type)> tensors)
        {
            foreach (var (name, _) in tensors)
            {
                if (name == "output.weight") return true;
                if (tiedEmbeddings && name == "token_embd.weight") return true;
            }
            return false;
        }

        var inPlaceItems = new List<(AblationSpec Spec, IReadOnlyList<(string TensorName, LlamaTensorType Type)> Tensors)>();
        var diskFallbackItems = new List<(AblationSpec Spec, IReadOnlyList<(string TensorName, LlamaTensorType Type)> Tensors)>();
        foreach (var item in allItems)
        {
            if (opts.DiskFallbackForLmHeadAblations && IsLmHeadCell(item.Tensors))
                diskFallbackItems.Add(item);
            else
                inPlaceItems.Add(item);
        }
        // Backwards-compat alias for existing channel-prime loop below.
        var workItems = inPlaceItems;

        // Unbounded channel — the work items are tiny (a list of tensor
        // names per cell), so memory cost of pre-loading all of them
        // is trivial and we avoid the back-pressure deadlock that a
        // bounded channel would create when the producer enqueues
        // before the worker pool exists.
        var channel = Channel.CreateUnbounded<
            (AblationSpec Spec, IReadOnlyList<(string TensorName, LlamaTensorType Type)> Tensors)>(
            new UnboundedChannelOptions
            {
                SingleReader = false,
                SingleWriter = true,
            });

        // Prime the channel synchronously — the unbounded variant of
        // WriteAsync never blocks, so this completes immediately and
        // workers see all 22 items waiting when they start draining.
        foreach (var item in workItems)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await channel.Writer.WriteAsync(item, cancellationToken).ConfigureAwait(false);
        }
        channel.Writer.Complete();

        async Task WorkerLoopAsync(int workerId)
        {
            // mmap MUST be off here: tied-embedding architectures
            // (Llama-3.2-1B, Qwen3-4B, etc.) keep token_embd.weight on a
            // CPU-resident buffer that, under mmap=true, points directly
            // at PROT_READ file pages. Any SetTensorData write to that
            // buffer SIGSEGVs the process. With mmap=false llama.cpp
            // copies tensor bytes into malloc'd pages at load time, so
            // the in-place ablator's writes are safe across all backends.
            // See issue #46.
            var modelOpts = new LlamaModelParameters { UseMmap = false, GpuLayerCount = -1 };
            using var model = new LlamaModel(sourceModelPath, modelOpts);
            using var ablator = new LlamaInPlaceAblator(model, sourceModelPath);

            await foreach (var (spec, tensors) in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                lock (sharedLock)
                {
                    progress?.Report(new Progress(
                        Stage.Quantizing, getScored(), totalJobs,
                        CurrentLabel: $"{spec.Target} @ {spec.Type} (in-place)",
                        CellTarget: spec.Target,
                        CellType: spec.Type,
                        CellState: CellState.Quantizing,
                        Fraction: fraction()));
                }

                var result = await ablator.RunAblationAsync(
                    tensors, corpusText, consumerPplOpts,
                    sharedImatrix, cancellationToken).ConfigureAwait(false);
                var ablation = result.Perplexity;

                lock (sharedLock)
                {
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
                    setScored(getScored() + 1);
                    progress?.Report(new Progress(
                        Stage.Scoring, getScored(), totalJobs,
                        CurrentLabel: $"{spec.Target} @ {spec.Type} = {ablation:F4}",
                        CellTarget: spec.Target,
                        CellType: spec.Type,
                        CellState: CellState.Done,
                        CellDelta: ablation - baseline,
                        Fraction: fraction()));
                }
            }
        }

        var workers = Enumerable.Range(0, pplConcurrency)
            .Select(WorkerLoopAsync)
            .ToArray();
        await Task.WhenAll(workers).ConfigureAwait(false);

        // Disk fallback for lm-head cells. Serial (one cell at a time):
        // they're a small minority (per-category mode: 1 cell per type
        // = 7 of 56; per-tensor mode: same single tensor across types).
        // Wall-time hit is bounded; correctness gain is large at low-bpw
        // budgets (issue #48).
        foreach (var (spec, tensors) in diskFallbackItems)
        {
            cancellationToken.ThrowIfCancellationRequested();
            lock (sharedLock)
            {
                progress?.Report(new Progress(
                    Stage.Quantizing, getScored(), totalJobs,
                    CurrentLabel: $"{spec.Target} @ {spec.Type} (disk fallback for lm-head)",
                    CellTarget: spec.Target,
                    CellType: spec.Type,
                    CellState: CellState.Quantizing,
                    Fraction: fraction()));
            }

            var slug = $"{Slugify(spec.Target)}_{spec.Type}_lmhead";
            var outPath = Path.Combine(workDir, $"{slug}.gguf");
            var recipe = BuildAblationRecipe(weightTensors, spec);
            await QuantizeAsync(
                sourceModelPath, outPath,
                ftype: LlamaFileType.Q4_K_M,
                imatrixPath: opts.ImatrixPath,
                recipe: recipe,
                cancellationToken).ConfigureAwait(false);

            double ablation;
            try
            {
                var modelOpts = new LlamaModelParameters { UseMmap = true, GpuLayerCount = -1 };
                using var model = new LlamaModel(outPath, modelOpts);
                var result = await LlamaPerplexity.ComputeAsync(
                    model, corpusText, consumerPplOpts,
                    progress: null, cancellationToken).ConfigureAwait(false);
                ablation = result.Perplexity;
            }
            finally
            {
                try { if (File.Exists(outPath)) File.Delete(outPath); }
                catch { /* best-effort cleanup; mirrors the disk path */ }
            }

            lock (sharedLock)
            {
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
                setScored(getScored() + 1);
                progress?.Report(new Progress(
                    Stage.Scoring, getScored(), totalJobs,
                    CurrentLabel: $"{spec.Target} @ {spec.Type} = {ablation:F4} (disk fallback)",
                    CellTarget: spec.Target,
                    CellType: spec.Type,
                    CellState: CellState.Done,
                    CellDelta: ablation - baseline,
                    Fraction: fraction()));
            }
        }
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

    /// <summary>
    /// Quantize <paramref name="source"/> to <paramref name="output"/> for
    /// either the F16 baseline (no recipe) or an ablation cell (recipe
    /// pinning one or more tensors to a candidate type, all others to F16).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why two paths.</b> The baseline (recipe == null) is a straight
    /// ftype=MostlyF16 quantize — every tensor lands at F16, the top of
    /// the ladder. <c>LlamaQuantizer</c>'s legacy override-only-elevates
    /// rule is vacuous here (no demotions to drop), so we use the
    /// straightforward path.
    /// </para>
    /// <para>
    /// The ablation path (recipe != null) <em>must</em> use
    /// <see cref="LlamaCustomQuantizer"/>. Run 14 documented that
    /// <see cref="LlamaQuantizer.QuantizeAsync"/> with
    /// <c>TensorTypeOverrides</c> silently drops a per-tensor override
    /// that would demote a tensor below the ftype heuristic's pick. The
    /// concrete consequence for ablation campaigns: stock Q4_K_M's
    /// <c>use_more_bits</c> heuristic puts <c>ffn_down</c> on certain
    /// layers at Q6_K. An ablation requesting Q4_K for one of those
    /// tensors would be silently kept at Q6_K — and the resulting GGUF
    /// would be bit-identical to the Q6_K ablation, producing identical
    /// PPL and bogus per-tensor data. The custom quantizer applies the
    /// recipe verbatim, so demotions are honored and per-tensor cells
    /// produce distinct files at distinct types.
    /// </para>
    /// </remarks>
    private static Task QuantizeAsync(
        string source, string output,
        LlamaFileType ftype,
        string? imatrixPath,
        LlamaQuantRecipe? recipe,
        CancellationToken ct)
    {
        if (recipe is not null)
        {
            var customOpts = new LlamaCustomQuantizerOptions
            {
                ImatrixPath = imatrixPath,
                // Heuristic skip list on: norms, expert-gate inputs, and
                // positional embeddings pass through at source type
                // unconditionally — same behavior as the legacy path's
                // tensor_allows_quantization filter, just on the custom
                // realizer.
                ApplyHeuristicSkipList = true,
            };
            return LlamaCustomQuantizer.QuantizeWithRecipeAsync(
                source, output, recipe, customOpts, progress: null, ct);
        }

        var p = new LlamaQuantizationParameters
        {
            FileType        = ftype,
            ImatrixPath     = imatrixPath,
            AllowRequantize = true,
        };
        return LlamaQuantizer.QuantizeAsync(source, output, p, ct);
    }
}
