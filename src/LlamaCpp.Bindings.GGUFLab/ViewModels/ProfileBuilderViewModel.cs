using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the Profile Builder page: run a per-category or per-layer
/// PPL ablation campaign that populates <see cref="LlamaInvestigationDb"/>
/// with measurement rows. Long-running — minutes for per-category on a
/// small model, days for per-layer on a 4B+ model.
/// </summary>
public sealed partial class ProfileBuilderViewModel : ToolPageViewModel
{
    public override string Title => "Profile Builder";
    public override string Description =>
        "Build a sensitivity profile: ablate each tensor (or category) at each candidate type, measure ΔPPL. Long-running — accumulates into the global investigation DB.";

    public enum CampaignMode
    {
        /// <summary>Cheap. Whole-category ablations, ~22 PPL runs by default.</summary>
        PerCategory,
        /// <summary>Expensive. One PPL run per (tensor, type). 28 layers × 7 cats × 7 types ≈ 1372 runs.</summary>
        PerLayer,
    }

    private readonly NativeLogBus _logBus;
    private readonly WorkspaceSettings _settings;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    [NotifyPropertyChangedFor(nameof(DiskEstimateText))]
    private string _sourceModelPath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowLoadWikitextRemedy))]
    private string _corpusPath = string.Empty;

    /// <summary>
    /// True when the user hasn't picked a corpus yet — surfaces the
    /// "load wikitext-2" remedy card, same pattern as the Imatrix /
    /// Perplexity / KL pages. Hidden as soon as a corpus path is set
    /// (own-data routes are always faster than the canonical fetch).
    /// </summary>
    public bool ShowLoadWikitextRemedy => string.IsNullOrEmpty(CorpusPath);

    [ObservableProperty]
    private string _imatrixPath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DiskEstimateText))]
    private string _workingDirectory = string.Empty;

    [ObservableProperty]
    private string _outputProfilePath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private int _contextSize = 512;

    [ObservableProperty]
    private double _kneeDeltaPplThreshold = 5.0;

    /// <summary>0 → auto.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DiskEstimateText))]
    private int _maxConcurrent;

    [ObservableProperty]
    private double _availableVramGb = 24.0;

    [ObservableProperty]
    private bool _cleanupWorkingDirectory = true;

    /// <summary>
    /// Campaign mode. Per-category is the cheap default; per-layer
    /// produces strictly more data (recipe builder demotes individual
    /// tensors that are below the category average) at proportionally
    /// higher compute cost.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPerCategory))]
    [NotifyPropertyChangedFor(nameof(IsPerLayer))]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private CampaignMode _mode = CampaignMode.PerCategory;

    public bool IsPerCategory => Mode == CampaignMode.PerCategory;
    public bool IsPerLayer    => Mode == CampaignMode.PerLayer;

    /// <summary>
    /// Categories with checkbox states. Auto-populated from
    /// <see cref="LlamaArchitectureRegistry"/> when the user selects
    /// a source model. Per-category mode only.
    /// </summary>
    public ObservableCollection<CategoryItem> Categories { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private string _detectedArchitecture = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private int _detectedLayerCount;

    /// <summary>
    /// Total parameter count of the source model (sum of tensor element
    /// counts). Drives the wall-time heuristic in the cost estimate.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private long _detectedParamCount;

    /// <summary>Tensor count auto-derived for per-layer mode (read-only).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private int _perLayerTensorCount;

    /// <summary>
    /// Comma-separated layer indices (with range support: "0-3, 5, 8, 20-27"),
    /// or empty = all layers. Per-layer mode only. Drives the targeted
    /// drill: combine with the categories selection to ablate just the
    /// (cat × layer) slice you care about — e.g. attn_v across all 36
    /// layers without touching ffn_down at all.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(EffectiveLayerCount))]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private string _layersText = string.Empty;

    /// <summary>How many layers <see cref="LayersText"/> resolves to (0 means "all").</summary>
    public int EffectiveLayerCount => ParseLayers().Count;

    // Candidate-type checkboxes — full ladder including the IQ family.
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useQ2K = true;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useIQ2S;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useQ3K;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useIQ3S;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useIQ4XS;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useQ4K = true;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useQ5K;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useQ6K = true;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private bool _useQ8_0;

    [ObservableProperty]
    private string _statusLine = "Idle. Pick a source model and corpus.";

    [ObservableProperty]
    private string _stageText = string.Empty;

    [ObservableProperty]
    private double _progressFraction;

    [ObservableProperty]
    private string _progressLabel = string.Empty;

    /// <summary>"03:42 elapsed" — refreshed by the dispatcher tick during a run.</summary>
    [ObservableProperty]
    private string _elapsedText = string.Empty;

    /// <summary>
    /// "ETA 12 min" — derived from <see cref="ProgressFraction"/> and
    /// elapsed once the campaign has visible progress. Empty until
    /// fraction crosses ~1% so the early estimate doesn't read as
    /// nonsense (e.g. "ETA 5 days" because we just started).
    /// </summary>
    [ObservableProperty]
    private string _etaText = string.Empty;

    /// <summary>
    /// Column headers for the progress grid (candidate-type names in
    /// ladder order). Populated from the campaign's Plan event so the
    /// grid shape stays stable across cell updates.
    /// </summary>
    public ObservableCollection<string> ProgressColumnHeaders { get; } = new();

    /// <summary>
    /// Rows of the progress grid. Each row covers one ablation target
    /// (a category or a tensor); cells are indexed by candidate type
    /// in <see cref="ProgressColumnHeaders"/> order. Allocated once
    /// from the Plan event; cell state updates flow through the row's
    /// observable cells without reshaping the grid.
    /// </summary>
    public ObservableCollection<ProgressRow> ProgressRows { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    /// <summary>
    /// Throttled, deduped, bounded log surface — same pattern as the
    /// Imatrix page. Native llama.cpp emits thousands of log lines per
    /// quantize/PPL run; the naive <c>StringBuilder + OnPropertyChanged</c>
    /// path forced a full TextBlock re-render per line and starved the
    /// UI thread of input events. This batches changes onto a 150 ms
    /// dispatcher tick, dedups consecutive identical lines, and caps
    /// visible runs while mirroring the full tail to a temp file.
    /// </summary>
    private readonly ThrottledLogBuffer _log = new();

    public string LogText => _log.Text;

    /// <summary>
    /// One-line cost estimate: "N measurements × type-count × target-count
    /// (≈ ~M cells)". Estimate refreshes on any input that affects N or
    /// the type/target counts.
    /// </summary>
    public string CostEstimateText
    {
        get
        {
            var typeCount = ResolveCandidateTypes().Count;
            if (typeCount == 0) return "No candidate types selected.";

            int targetCount;
            string targetLabel;
            if (Mode == CampaignMode.PerCategory)
            {
                targetCount = Categories.Count(c => c.IsSelected);
                targetLabel = "categories";
                if (targetCount == 0)
                    return "No categories selected.";
            }
            else
            {
                // Per-layer with optional drill: the actual ablation
                // count is (selected categories × selected layers) + any
                // top-level entries the user kept ticked.
                targetCount = ResolvePerLayerTargetTensorCount();
                targetLabel = "tensors";
                if (targetCount == 0)
                    return "Pick categories and (optionally) layers to drill into.";
            }
            int total = 1 + targetCount * typeCount;    // +1 for baseline
            var layerNote = Mode == CampaignMode.PerLayer && EffectiveLayerCount > 0
                ? $", layers={EffectiveLayerCount}"
                : "";
            var timeNote = EstimateWallTimeText(total) is { } te
                ? $"  ·  estimated time: {te}"
                : "";
            return $"≈ {total} PPL runs ({targetCount} {targetLabel} × {typeCount} types + 1 baseline{layerNote}){timeNote}. " +
                   "Already-measured cells in the DB are skipped.";
        }
    }

    /// <summary>
    /// Rough wall-time estimate based on a (param-count × context-size)
    /// heuristic calibrated against the runs we have data on:
    /// 0.6B/512 ≈ ~16 s/cell, 1.7B/512 ≈ ~50 s/cell, 4B/512 ≈ ~120 s/cell.
    /// Returns null when paramCount or ctx aren't known yet — caller
    /// hides the time note in that case.
    /// </summary>
    /// <remarks>
    /// This is a forward-projection heuristic, not a measurement. The
    /// actual wall time depends on GPU concurrency (the parallel PPL
    /// runner is the dominant factor on big-VRAM hardware), file I/O,
    /// and PPL chunk count from the corpus. Treat the estimate as an
    /// upper-bound rule of thumb.
    /// </remarks>
    private string? EstimateWallTimeText(int totalCells)
    {
        if (DetectedParamCount <= 0 || ContextSize <= 0 || totalCells <= 0) return null;
        var paramsB = DetectedParamCount / 1e9;
        // 30 s/cell at 1 B params and ctx=512 — fits the empirical
        // 1.7B/512 ≈ 50 s and 4B/512 ≈ 120 s observations.
        var perCellSeconds = 30.0 * paramsB * (ContextSize / 512.0);
        var totalSeconds = perCellSeconds * totalCells;
        return FormatDuration(totalSeconds);
    }

    /// <summary>
    /// Estimated peak disk usage during a run plus available free
    /// space on the working directory's partition. Drives both the
    /// pre-build display and the pre-flight check that refuses Build
    /// when the estimate exceeds available.
    /// </summary>
    /// <remarks>
    /// Each in-flight quantize file is roughly the size of the F16
    /// source (ablations leave ~all weights at F16, so output ≈
    /// source size). The builder's continuous-flow pipeline keeps at
    /// most <c>1 (producer mid-write) + 1 (channel) + pplConcurrency
    /// (consumers actively scoring)</c> files on disk simultaneously,
    /// so peak = <c>(2 + pplConcurrency) × sourceSize</c>.
    /// pplConcurrency mirrors the builder's resolution: explicit
    /// <see cref="MaxConcurrent"/> if set, otherwise <c>min(ProcessorCount, 8)</c>
    /// as an upper-bound estimate (the runtime VRAM-aware heuristic
    /// may pick lower, which only reduces peak).
    /// </remarks>
    public string DiskEstimateText
    {
        get
        {
            if (string.IsNullOrEmpty(SourceModelPath) || !File.Exists(SourceModelPath)) return string.Empty;

            long sourceSize;
            try { sourceSize = new FileInfo(SourceModelPath).Length; }
            catch { return string.Empty; }

            long peak = (long)(2 + EstimatedPplConcurrency()) * sourceSize;

            var workDir = string.IsNullOrEmpty(WorkingDirectory) ? Path.GetTempPath() : WorkingDirectory;
            var free = GetFreeBytes(workDir);

            string peakText = FormatBytes(peak);
            string freeText = free is long f ? FormatBytes(f) : "unknown";
            string warn = free is long f2 && peak > f2
                ? "  ·  ⚠ insufficient — Build will refuse"
                : "";
            return $"peak disk in flight: ~{peakText}  ·  free at {Path.GetFileName(workDir.TrimEnd(Path.DirectorySeparatorChar))}: {freeText}{warn}";
        }
    }

    /// <summary>True when the working-directory partition has enough headroom for the estimated peak.</summary>
    public bool DiskEstimateIsSufficient
    {
        get
        {
            if (string.IsNullOrEmpty(SourceModelPath) || !File.Exists(SourceModelPath)) return true;
            long sourceSize;
            try { sourceSize = new FileInfo(SourceModelPath).Length; }
            catch { return true; }
            long peak = (long)(2 + EstimatedPplConcurrency()) * sourceSize;
            var workDir = string.IsNullOrEmpty(WorkingDirectory) ? Path.GetTempPath() : WorkingDirectory;
            var free = GetFreeBytes(workDir);
            return free is null || free.Value >= peak;
        }
    }

    private int EstimatedPplConcurrency() => MaxConcurrent > 0
        ? MaxConcurrent
        : Math.Min(Environment.ProcessorCount, 8);

    /// <summary>
    /// Free bytes on the partition containing <paramref name="path"/>.
    /// Walks <see cref="DriveInfo.GetDrives"/> finding the longest mount
    /// point that prefixes the absolute path, so it works on Linux
    /// where /tmp may be a separate mount or part of /. Returns null
    /// when the lookup fails.
    /// </summary>
    private static long? GetFreeBytes(string path)
    {
        try
        {
            if (string.IsNullOrEmpty(path)) return null;
            // Resolve to an absolute path; create the directory if needed
            // to make DriveInfo work for paths that don't exist yet.
            var full = Path.GetFullPath(path);
            if (!Directory.Exists(full))
            {
                // Probe the closest existing ancestor — don't create the
                // directory just to ask about disk space.
                var probe = full;
                while (!string.IsNullOrEmpty(probe) && !Directory.Exists(probe))
                    probe = Path.GetDirectoryName(probe);
                if (string.IsNullOrEmpty(probe)) return null;
                full = probe;
            }

            DriveInfo? best = null;
            foreach (var d in DriveInfo.GetDrives())
            {
                if (!d.IsReady) continue;
                if (full.StartsWith(d.RootDirectory.FullName, StringComparison.Ordinal))
                {
                    if (best is null || d.RootDirectory.FullName.Length > best.RootDirectory.FullName.Length)
                        best = d;
                }
            }
            return best?.AvailableFreeSpace;
        }
        catch { return null; }
    }

    /// <summary>Format a byte count compactly: 512 B / 13.4 KB / 2.31 GB.</summary>
    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes} B";
        double v = bytes;
        string[] units = { "KB", "MB", "GB", "TB" };
        int i = -1;
        while (v >= 1024 && i < units.Length - 1) { v /= 1024; i++; }
        return $"{v:F2} {units[i]}";
    }

    /// <summary>Compact human duration: 45s / 12m / 2h 15m / 3d 4h.</summary>
    private static string FormatDuration(double seconds)
    {
        if (seconds < 60) return $"{seconds:F0} s";
        if (seconds < 3600)
        {
            var m = seconds / 60.0;
            return $"{m:F0} min";
        }
        if (seconds < 86400)
        {
            int h = (int)(seconds / 3600);
            int m = (int)((seconds % 3600) / 60);
            return m > 0 ? $"{h} h {m} min" : $"{h} h";
        }
        int d = (int)(seconds / 86400);
        int rh = (int)((seconds % 86400) / 3600);
        return rh > 0 ? $"{d} d {rh} h" : $"{d} d";
    }

    /// <summary>
    /// How many tensors a per-layer campaign would actually ablate
    /// given the current category and layer selection.
    /// </summary>
    private int ResolvePerLayerTargetTensorCount()
    {
        if (string.IsNullOrEmpty(DetectedArchitecture)) return 0;
        if (DetectedLayerCount <= 0) return 0;
        var spec = LlamaArchitectureRegistry.Lookup(DetectedArchitecture)
                ?? LlamaArchitectureRegistry.StandardTransformer;
        var selectedCats = Categories.Where(c => c.IsSelected).Select(c => c.Name).ToList();
        if (selectedCats.Count == 0) return 0;
        var layers = ParseLayers();
        var layerFilter = layers.Count == 0 ? null : layers;
        return spec.ResolveTensors(DetectedLayerCount, selectedCats, layerFilter).Count();
    }

    private CancellationTokenSource? _cts;

    /// <summary>
    /// Row count in the investigation DB matching the current campaign
    /// configuration (model + corpus + imatrix + ctx + selected
    /// categories × layers × types). Updated lazily by
    /// <see cref="RefreshDbRowCountsCommand"/> — the count requires
    /// content-hashing the source/imatrix files and reading the corpus,
    /// which we don't want to redo on every keystroke.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasMatchingDbRows))]
    private long _dbRowCountMatchingSelection;

    /// <summary>Total rows in the DB matching this campaign signature, regardless of selection.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasCampaignRows))]
    private long _dbRowCountForCampaign;

    public bool HasMatchingDbRows => DbRowCountMatchingSelection > 0;
    public bool HasCampaignRows   => DbRowCountForCampaign       > 0;

    [ObservableProperty]
    private string _dbStatusLine = "Click 'Refresh DB counts' to see what's already measured.";

    public ProfileBuilderViewModel(NativeLogBus logBus, WorkspaceSettings settings)
    {
        _logBus = logBus;
        _settings = settings;

        // Hydrate from settings so the working directory persists across
        // app restarts. Empty string = fall back to system temp inside
        // the builder (matches the field's pre-settings semantics).
        WorkingDirectory = settings.ProfileBuilderScratchDirectory ?? string.Empty;

        Categories.CollectionChanged += (_, _) => OnPropertyChanged(nameof(CostEstimateText));
        _log.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(ThrottledLogBuffer.Text))
                OnPropertyChanged(nameof(LogText));
        };
    }

    /// <summary>
    /// Fetch (or read from cache) the standard wikitext-2 raw test set
    /// and assign it as the calibration corpus. Same pattern as the
    /// Imatrix / Perplexity / KL Divergence pages — first call downloads
    /// (~700 KB), subsequent calls hit the cache.
    /// </summary>
    [RelayCommand]
    private async Task LoadWikitextAsync()
    {
        if (IsRunning) return;
        try
        {
            StatusLine = WikitextCorpus.IsCached()
                ? "Loading cached wikitext-2 test set…"
                : "Downloading wikitext-2 test set from HuggingFace (~700 KB)…";

            var progress = new Progress<(long downloaded, long? total)>(p =>
            {
                if (p.total is long t && t > 0)
                {
                    int pct = (int)(100L * p.downloaded / t);
                    StatusLine = $"Downloading wikitext-2 test set… {p.downloaded / 1024:N0} / {t / 1024:N0} KB ({pct}%)";
                }
                else
                {
                    StatusLine = $"Downloading wikitext-2 test set… {p.downloaded / 1024:N0} KB";
                }
            });

            var path = await WikitextCorpus.EnsureTestRawAsync(progress);
            CorpusPath = path;
            StatusLine = $"Loaded wikitext-2 test corpus from {Path.GetFileName(path)}.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to load wikitext-2: {ex.Message}";
        }
    }

    [RelayCommand]
    private async Task BuildAsync()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(SourceModelPath) || !File.Exists(SourceModelPath))
        {
            StatusLine = "Pick a valid source GGUF (F16 recommended).";
            return;
        }
        if (string.IsNullOrWhiteSpace(CorpusPath) || !File.Exists(CorpusPath))
        {
            StatusLine = "Pick a calibration corpus (e.g. wiki.test.raw).";
            return;
        }
        var candidates = ResolveCandidateTypes();
        if (candidates.Count == 0)
        {
            StatusLine = "Pick at least one candidate type.";
            return;
        }

        IReadOnlyList<string> selectedCategories = Array.Empty<string>();
        if (Mode == CampaignMode.PerCategory)
        {
            selectedCategories = Categories.Where(c => c.IsSelected).Select(c => c.Name).ToList();
            if (selectedCategories.Count == 0)
            {
                StatusLine = "Pick at least one category.";
                return;
            }
        }

        if (string.IsNullOrWhiteSpace(OutputProfilePath))
        {
            try
            {
                var dir = Path.GetDirectoryName(SourceModelPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(SourceModelPath);
                OutputProfilePath = Path.Combine(dir, $"{stem}.profile.json");
            }
            catch
            {
                StatusLine = "Pick an output profile path.";
                return;
            }
        }

        // Pre-flight disk check. The campaign's continuous-flow pipeline
        // holds up to (2 + pplConcurrency) ~F16-sized files in flight
        // (1 producer mid-write + 1 channel + N consumers scoring); if
        // the working directory's partition can't fit that, fail before
        // quantizing rather than mid-campaign with a half-full /tmp.
        if (!DiskEstimateIsSufficient)
        {
            StatusLine = $"Insufficient disk for working directory. {DiskEstimateText}. " +
                         "Set 'Working dir' to a partition with enough headroom.";
            return;
        }

        _log.Clear();
        ProgressFraction = 0;
        ProgressLabel = string.Empty;
        StageText = "starting…";
        ElapsedText = string.Empty;
        EtaText = string.Empty;
        _cts = new CancellationTokenSource();
        IsRunning = true;
        var startedAt = DateTime.Now;
        StartElapsedTimer(startedAt);
        StatusLine = Mode == CampaignMode.PerCategory
            ? "Building per-category profile…"
            : "Building per-layer profile…";

        // Subscribe through the throttled buffer instead of a raw
        // StringBuilder + per-line PropertyChanged — native llama.cpp
        // log volume during quantize/PPL is high enough to starve UI
        // input handling otherwise (mouse hover, scroll wheel).
        var unsubscribe = _logBus.Subscribe(line => _log.Append(line));

        try
        {
            var opts = new LlamaSensitivityProfileBuilder.Options
            {
                CandidateTypes          = candidates,
                Categories              = selectedCategories.Count > 0 ? selectedCategories : new[] { "ffn_up" },
                ImatrixPath             = string.IsNullOrWhiteSpace(ImatrixPath) ? null : ImatrixPath,
                MaxConcurrent           = MaxConcurrent,
                AvailableVramBytes      = AvailableVramGb > 0 ? (long)(AvailableVramGb * 1024 * 1024 * 1024) : null,
                WorkingDirectory        = string.IsNullOrWhiteSpace(WorkingDirectory) ? null : WorkingDirectory,
                CleanupWorkingDirectory = CleanupWorkingDirectory,
                KneeDeltaPplThreshold   = KneeDeltaPplThreshold,
                PerplexityOptions       = new LlamaPerplexityOptions { ContextSize = ContextSize },
            };
            var progress = new Progress<LlamaSensitivityProfileBuilder.Progress>(p =>
            {
                StageText = p.Stage.ToString();
                // Bind the bar to the library's monotonic Fraction so it
                // never rewinds when a batch flips from quantize to score.
                // p.CompletedJobs still counts whole cells (post-PPL),
                // which is the more meaningful number for the label.
                ProgressFraction = p.Fraction;
                ProgressLabel = $"{p.CompletedJobs}/{p.TotalJobs}" +
                    (string.IsNullOrEmpty(p.CurrentLabel) ? "" : $"  ·  {p.CurrentLabel}");

                // Plan event: build the grid shape once, before any cell updates.
                if (p.Stage == LlamaSensitivityProfileBuilder.Stage.Plan && p.Plan is { } plan)
                {
                    AllocateProgressGrid(plan);
                    return;
                }

                // Per-cell update: locate the cell in the existing grid
                // and update only its state/delta. The grid shape is
                // unchanged.
                if (p.CellTarget is { } target && p.CellType is { } type && p.CellState is { } state)
                {
                    UpdateProgressCell(target, type, state, p.CellDelta);
                }
            });

            // Wrap the whole call in Task.Run so the synchronous prefix
            // of BuildAsync (GGUF open, tensor enumeration, identity
            // hashing of multi-GB files) doesn't block the UI thread.
            // The async method's internal awaits are already correctly
            // off-thread, but the work *before* the first await runs
            // on the caller's thread — which is the UI dispatcher when
            // invoked from a RelayCommand. progress.Report still
            // dispatches back to the UI thread via its captured
            // SynchronizationContext, so binding updates are unaffected.
            if (Mode == CampaignMode.PerCategory)
            {
                var src = SourceModelPath;
                var corp = CorpusPath;
                var optsLocal = opts;
                var ct = _cts.Token;
                var profile = await Task.Run(
                    () => LlamaSensitivityProfileBuilder.BuildAsync(src, corp, optsLocal, progress, ct),
                    ct);

                profile.SaveToJson(OutputProfilePath);
                var elapsed = DateTime.Now - startedAt;
                StatusLine =
                    $"Wrote {OutputProfilePath} in {elapsed.TotalMinutes:F1} min " +
                    $"(arch={profile.ArchitectureId}, layers={profile.LayerCount}, F16 PPL={profile.F16BaselinePerplexity:F3}).";
            }
            else
            {
                var spec = LlamaArchitectureRegistry.Lookup(DetectedArchitecture)
                        ?? LlamaArchitectureRegistry.StandardTransformer;
                var selectedCats = Categories.Where(c => c.IsSelected).Select(c => c.Name).ToList();
                var layers = ParseLayers();
                var layerFilter = layers.Count == 0 ? null : layers;
                var targetTensors = spec.ResolveTensors(
                    Math.Max(1, DetectedLayerCount), selectedCats, layerFilter).ToList();

                var src = SourceModelPath;
                var corp = CorpusPath;
                var optsLocal = opts;
                var ct = _cts.Token;
                var targets = targetTensors;
                var newCount = await Task.Run(
                    () => LlamaSensitivityProfileBuilder.BuildPerLayerAsync(src, corp, targets, optsLocal, progress, ct),
                    ct);
                var elapsed = DateTime.Now - startedAt;
                StatusLine =
                    $"Recorded {newCount} new per-tensor measurements in {elapsed.TotalMinutes:F1} min " +
                    "(data is in the investigation DB; the apply page reads it via DeriveFromDb).";
            }
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled. Measurements taken so far are preserved in the DB — re-run to resume.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Build failed: {ex.Message}";
            _log.Append($"[error] {ex}");
        }
        finally
        {
            unsubscribe();
            _log.Stop();    // flush any pending tick before idle
            StopElapsedTimer();
            IsRunning = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    /// <summary>
    /// Spin up a 1 Hz dispatcher tick that refreshes <see cref="ElapsedText"/>
    /// and (when there's enough progress) <see cref="EtaText"/>. Cheap —
    /// we just format two strings per second; the underlying counters
    /// already update from the campaign's progress events.
    /// </summary>
    private void StartElapsedTimer(DateTime startedAt)
    {
        StopElapsedTimer();
        _elapsedTimer = new Avalonia.Threading.DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(1),
        };
        _elapsedTimer.Tick += (_, _) =>
        {
            var elapsed = (DateTime.Now - startedAt).TotalSeconds;
            ElapsedText = FormatDuration(elapsed);
            // Wait until ~1% progress before showing an ETA — earlier
            // estimates from a tiny fraction blow up to "ETA 5 days"
            // and just confuse the user.
            if (ProgressFraction > 0.01)
            {
                var remaining = elapsed * (1.0 / ProgressFraction - 1.0);
                EtaText = $"ETA {FormatDuration(remaining)} remaining";
            }
        };
        _elapsedTimer.Start();
    }

    private void StopElapsedTimer()
    {
        _elapsedTimer?.Stop();
        _elapsedTimer = null;
    }

    private Avalonia.Threading.DispatcherTimer? _elapsedTimer;

    [RelayCommand]
    private void Cancel()
    {
        _cts?.Cancel();
        StatusLine = IsRunning ? "Cancellation requested…" : StatusLine;
    }

    private IReadOnlyList<LlamaTensorType> ResolveCandidateTypes()
    {
        // Order: ascending bpw (matches the rest of the codebase's
        // convention for ladder construction).
        var picks = new List<LlamaTensorType>();
        if (UseIQ2S)   picks.Add(LlamaTensorType.IQ2_S);
        if (UseQ2K)    picks.Add(LlamaTensorType.Q2_K);
        if (UseIQ3S)   picks.Add(LlamaTensorType.IQ3_S);
        if (UseQ3K)    picks.Add(LlamaTensorType.Q3_K);
        if (UseIQ4XS)  picks.Add(LlamaTensorType.IQ4_XS);
        if (UseQ4K)    picks.Add(LlamaTensorType.Q4_K);
        if (UseQ5K)    picks.Add(LlamaTensorType.Q5_K);
        if (UseQ6K)    picks.Add(LlamaTensorType.Q6_K);
        if (UseQ8_0)   picks.Add(LlamaTensorType.Q8_0);
        return picks;
    }

    /// <summary>
    /// Read the source GGUF, look up its architecture in the registry,
    /// and populate <see cref="Categories"/> + per-layer tensor count.
    /// Resilient to bad paths — silently no-ops if the model can't be
    /// opened (the user gets feedback through the status line on
    /// other paths).
    /// </summary>
    private void RefreshArchitectureFromSource()
    {
        Categories.Clear();
        DetectedArchitecture = string.Empty;
        DetectedLayerCount = 0;
        DetectedParamCount = 0;
        PerLayerTensorCount = 0;

        if (string.IsNullOrWhiteSpace(SourceModelPath) || !File.Exists(SourceModelPath)) return;

        try
        {
            var gguf = LlamaGgufFile.Open(SourceModelPath);
            var archEntry = gguf.Metadata.FirstOrDefault(m => m.Key == "general.architecture");
            var archId = archEntry?.Value.AsString() ?? "unknown";
            DetectedArchitecture = archId;

            var layerEntry = gguf.Metadata.FirstOrDefault(m => m.Key == $"{archId}.block_count");
            DetectedLayerCount = layerEntry?.Value.Type switch
            {
                LlamaGgufType.Uint32 => (int)layerEntry.Value.AsUInt32(),
                LlamaGgufType.Int32  => layerEntry.Value.AsInt32(),
                LlamaGgufType.Uint64 => (int)layerEntry.Value.AsUInt64(),
                _                    => 0,
            };

            // Total parameter count = sum of element counts across all
            // tensors. Drives the wall-time heuristic for the cost
            // estimate; matches the convention LlamaSensitivityProfileBuilder
            // uses for the same purpose.
            DetectedParamCount = gguf.Tensors.Sum(t => t.Dimensions.Aggregate(1L, (a, b) => a * (long)b));

            var spec = LlamaArchitectureRegistry.Lookup(archId)
                    ?? LlamaArchitectureRegistry.StandardTransformer;

            // Filter to categories that actually have at least one
            // matching tensor in this specific model. Catches tied-
            // embedding models (no output.weight) automatically.
            var tensorNames = gguf.Tensors
                .Where(t => t.Dimensions.Length > 1 && t.Name.EndsWith(".weight"))
                .Select(t => t.Name)
                .ToList();

            foreach (var cat in spec.Categories)
            {
                if (!tensorNames.Any(n => CategoryMatch(n, cat))) continue;
                Categories.Add(new CategoryItem(cat, isSelected: true) { Owner = this });
            }

            // Per-layer count: union of expanded per-layer + top-level,
            // then filter to tensors actually present.
            if (DetectedLayerCount > 0)
            {
                var present = new HashSet<string>(tensorNames, StringComparer.Ordinal);
                var perLayer = spec.ExpandPerLayerTensors(DetectedLayerCount).Where(present.Contains);
                var topLevel = spec.TopLevelTensors.Where(present.Contains);
                PerLayerTensorCount = perLayer.Count() + topLevel.Count();
            }
        }
        catch
        {
            // Best-effort — leave the form empty if we can't read the file.
        }
    }

    /// <summary>
    /// Parse <see cref="LayersText"/> into a sorted, deduplicated list
    /// of layer indices. Accepts comma-separated indices and ranges
    /// (<c>"0-3, 5, 8, 20-27"</c>). Empty input → empty list, which
    /// the campaign treats as "all layers."
    /// </summary>
    public IReadOnlyList<int> ParseLayers()
    {
        if (string.IsNullOrWhiteSpace(LayersText)) return Array.Empty<int>();
        var result = new HashSet<int>();
        foreach (var raw in LayersText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var dash = raw.IndexOf('-');
            if (dash > 0 && int.TryParse(raw[..dash].Trim(), out var lo)
                         && int.TryParse(raw[(dash + 1)..].Trim(), out var hi))
            {
                if (lo > hi) (lo, hi) = (hi, lo);
                for (int i = lo; i <= hi; i++) result.Add(i);
            }
            else if (int.TryParse(raw, out var n))
            {
                result.Add(n);
            }
        }
        return result.OrderBy(i => i).ToList();
    }

    /// <summary>Preset: all layers (clears <see cref="LayersText"/>, which the campaign treats as "all").</summary>
    [RelayCommand]
    private void PresetLayersAll() => LayersText = string.Empty;

    /// <summary>Preset: first 4 layers.</summary>
    [RelayCommand]
    private void PresetLayersFirst4()
    {
        var n = Math.Min(4, Math.Max(0, DetectedLayerCount));
        LayersText = n > 0 ? string.Join(",", Enumerable.Range(0, n)) : string.Empty;
    }

    /// <summary>Preset: last 4 layers.</summary>
    [RelayCommand]
    private void PresetLayersLast4()
    {
        if (DetectedLayerCount <= 0) { LayersText = string.Empty; return; }
        var start = Math.Max(0, DetectedLayerCount - 4);
        LayersText = string.Join(",", Enumerable.Range(start, DetectedLayerCount - start));
    }

    /// <summary>
    /// Preset: stock's <see cref="LlamaStockBaseline.UseMoreBits"/>
    /// alternation set — the layers llama.cpp protects with extra bpw
    /// in its hand-tuned heuristic. Drilling into this set vs the rest
    /// is the empirical validation we want for "is the heuristic right?"
    /// </summary>
    [RelayCommand]
    private void PresetLayersUseMoreBits()
    {
        if (DetectedLayerCount <= 0) { LayersText = string.Empty; return; }
        var picks = Enumerable.Range(0, DetectedLayerCount)
            .Where(i => LlamaStockBaseline.UseMoreBits(i, DetectedLayerCount))
            .ToList();
        LayersText = string.Join(",", picks);
    }

    /// <summary>Preset: stratified sample — first 1, last 1, plus ~3-5 evenly-spaced middle layers.</summary>
    [RelayCommand]
    private void PresetLayersStratified()
    {
        if (DetectedLayerCount <= 0) { LayersText = string.Empty; return; }
        var n = DetectedLayerCount;
        var picks = new SortedSet<int> { 0, n - 1 };
        // 5-layer total target keeps cost tractable on big models.
        for (int k = 1; k <= 3; k++)
            picks.Add(k * n / 4);
        LayersText = string.Join(",", picks);
    }

    private static bool CategoryMatch(string tensorName, string category) =>
        category.Contains('.')
            ? tensorName == category ||
              tensorName.EndsWith("." + category, StringComparison.Ordinal)
            : tensorName.Contains(category, StringComparison.Ordinal);

    /// <summary>
    /// Build the progress grid shape from the campaign's Plan event:
    /// one row per distinct target, one column per distinct candidate
    /// type. Allocated once; subsequent cell updates only touch
    /// observable cell properties so the layout never reshapes.
    /// </summary>
    private void AllocateProgressGrid(IReadOnlyList<(string Target, LlamaTensorType Type)> plan)
    {
        ProgressColumnHeaders.Clear();
        ProgressRows.Clear();

        var distinctTypes = plan.Select(p => p.Type).Distinct().OrderBy(t => t).ToList();
        foreach (var t in distinctTypes)
            ProgressColumnHeaders.Add(t.ToString());

        var rowsByTarget = new Dictionary<string, ProgressRow>(StringComparer.Ordinal);
        var typeIndex = distinctTypes
            .Select((t, i) => (t, i))
            .ToDictionary(x => x.t, x => x.i);

        foreach (var (target, _) in plan)
        {
            if (!rowsByTarget.TryGetValue(target, out var row))
            {
                row = new ProgressRow(DisplayLabel(target), distinctTypes.Count);
                rowsByTarget[target] = row;
                ProgressRows.Add(row);
            }
        }

        // Pre-fill cells with their type so the row's cell index lines
        // up with ProgressColumnHeaders. Each cell starts Pending; the
        // builder's Resumed events bring already-measured cells up to
        // their state immediately.
        foreach (var (target, type) in plan)
        {
            var row = rowsByTarget[target];
            var idx = typeIndex[type];
            if (row.Cells[idx] is null)
                row.Cells[idx] = new ProgressCell(type) { Owner = this };
        }
    }

    /// <summary>
    /// Find the named cell in the grid and update its state + delta.
    /// No-op if the cell isn't in the grid (e.g. an update arrives
    /// before the Plan event, which shouldn't happen but is harmless).
    /// </summary>
    private void UpdateProgressCell(string target, LlamaTensorType type, LlamaSensitivityProfileBuilder.CellState state, double? delta)
    {
        var label = DisplayLabel(target);
        var row = ProgressRows.FirstOrDefault(r => r.Label == label);
        if (row is null) return;
        var cell = row.Cells.FirstOrDefault(c => c is not null && c.Type == type);
        if (cell is null) return;
        cell.State = state;
        if (delta is double d) cell.Delta = d;
    }

    /// <summary>
    /// Compact display label for an ablation target. "category:ffn_up" → "ffn_up";
    /// "tensor:blk.13.attn_v.weight" → "blk.13.attn_v" (trailing .weight is implied).
    /// </summary>
    private static string DisplayLabel(string target)
    {
        if (target.StartsWith("category:", StringComparison.Ordinal))
            return target["category:".Length..];
        if (target.StartsWith("tensor:", StringComparison.Ordinal))
        {
            var name = target["tensor:".Length..];
            if (name.EndsWith(".weight", StringComparison.Ordinal))
                name = name[..^".weight".Length];
            return name;
        }
        return target;
    }

    /// <summary>
    /// Compute the campaign identity tuple (model_sha, corpus_sha,
    /// imatrix_sha, ctx) for the current form state, plus the resolved
    /// list of ablation targets and types from the user's selections.
    /// Returns null when any input is missing — callers display an
    /// "incomplete inputs" state in that case.
    /// </summary>
    /// <remarks>
    /// Hashing the source GGUF and (optional) imatrix files reads ~2 MB
    /// each (head + tail samples); reading the corpus is the full file.
    /// Run on a background task — the DB controls explicitly trigger
    /// this rather than recomputing on every form change.
    /// </remarks>
    private async Task<DbScope?> ResolveDbScopeAsync(CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(SourceModelPath) || !File.Exists(SourceModelPath)) return null;
        if (string.IsNullOrWhiteSpace(CorpusPath) || !File.Exists(CorpusPath)) return null;

        return await Task.Run(() =>
        {
            var modelSha   = LlamaInvestigationDb.ComputeContentSha(SourceModelPath);
            var corpusText = File.ReadAllText(CorpusPath);
            var corpusSha  = LlamaInvestigationDb.ComputeTextSha(corpusText);
            var imatrixSha = string.IsNullOrEmpty(ImatrixPath) || !File.Exists(ImatrixPath)
                ? LlamaInvestigationDb.NoImatrixSha
                : LlamaInvestigationDb.ComputeContentSha(ImatrixPath);

            // Resolve the targets from the current category × layer ×
            // mode selection, matching the same logic BuildAsync uses
            // when the campaign actually runs.
            var targets = ResolveCurrentTargets();
            var types   = ResolveCandidateTypes();
            return new DbScope(modelSha, corpusSha, imatrixSha, ContextSize, targets, types);
        }, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Build the list of ablation targets the current selection would
    /// produce. Per-category mode produces "category:&lt;name&gt;" entries;
    /// per-layer mode expands the architecture spec across selected
    /// categories and layers.
    /// </summary>
    private IReadOnlyList<string> ResolveCurrentTargets()
    {
        var selectedCats = Categories.Where(c => c.IsSelected).Select(c => c.Name).ToList();
        if (selectedCats.Count == 0) return Array.Empty<string>();

        if (Mode == CampaignMode.PerCategory)
            return selectedCats.Select(c => $"category:{c}").ToList();

        // Per-layer: same call BuildAsync uses, scoped to current selection.
        var spec = LlamaArchitectureRegistry.Lookup(DetectedArchitecture)
                ?? LlamaArchitectureRegistry.StandardTransformer;
        var layers = ParseLayers();
        var layerFilter = layers.Count == 0 ? null : layers;
        return spec.ResolveTensors(Math.Max(1, DetectedLayerCount), selectedCats, layerFilter)
            .Select(t => $"tensor:{t}")
            .ToList();
    }

    /// <summary>
    /// Refresh the two DB row-count properties from the live DB. Call
    /// this from the UI before previewing or invoking the invalidation
    /// commands — the counts make the buttons' effects visible without
    /// re-running the campaign.
    /// </summary>
    [RelayCommand]
    private async Task RefreshDbRowCountsAsync()
    {
        try
        {
            var scope = await ResolveDbScopeAsync();
            if (scope is null)
            {
                DbRowCountMatchingSelection = 0;
                DbRowCountForCampaign = 0;
                DbStatusLine = "Need source GGUF + corpus to inspect the DB.";
                return;
            }

            await Task.Run(() =>
            {
                using var db = LlamaInvestigationDb.Open();
                var campaign = db.CountMatching(scope.ModelSha, scope.CorpusSha, scope.ImatrixSha, scope.ContextSize);
                var selection = scope.Targets.Count == 0 || scope.Types.Count == 0
                    ? 0L
                    : db.CountMatching(scope.ModelSha, scope.CorpusSha, scope.ImatrixSha, scope.ContextSize,
                        scope.Targets, scope.Types);
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    DbRowCountMatchingSelection = selection;
                    DbRowCountForCampaign = campaign;
                    DbStatusLine = $"DB has {campaign} rows for this campaign signature " +
                                   $"({selection} match the current category × layer × type selection).";
                });
            });
        }
        catch (Exception ex)
        {
            DbStatusLine = $"DB inspect failed: {ex.Message}";
        }
    }

    /// <summary>
    /// Delete the rows that the current category × layer × type selection
    /// would re-measure on the next Build. The next campaign will run
    /// every previously-deleted cell from scratch; cells outside the
    /// selection are preserved.
    /// </summary>
    [RelayCommand]
    private async Task InvalidateMatchingSelectionAsync()
    {
        try
        {
            var scope = await ResolveDbScopeAsync();
            if (scope is null || scope.Targets.Count == 0 || scope.Types.Count == 0)
            {
                DbStatusLine = "Need source GGUF + corpus + at least one selected category and type.";
                return;
            }

            await Task.Run(() =>
            {
                using var db = LlamaInvestigationDb.Open();
                var deleted = db.DeleteMatching(
                    scope.ModelSha, scope.CorpusSha, scope.ImatrixSha, scope.ContextSize,
                    scope.Targets, scope.Types);
                var remaining = db.CountMatching(scope.ModelSha, scope.CorpusSha, scope.ImatrixSha, scope.ContextSize);
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    DbRowCountMatchingSelection = 0;
                    DbRowCountForCampaign = remaining;
                    DbStatusLine = $"Deleted {deleted} matching rows. {remaining} rows remain for this campaign signature.";
                });
            });
        }
        catch (Exception ex)
        {
            DbStatusLine = $"Invalidation failed: {ex.Message}";
        }
    }

    /// <summary>
    /// Delete every row for this campaign's identity tuple — wipes the
    /// baseline along with all ablation cells. The next campaign runs
    /// from scratch.
    /// </summary>
    [RelayCommand]
    private async Task InvalidateAllForCampaignAsync()
    {
        try
        {
            var scope = await ResolveDbScopeAsync();
            if (scope is null)
            {
                DbStatusLine = "Need source GGUF + corpus to identify the campaign.";
                return;
            }

            await Task.Run(() =>
            {
                using var db = LlamaInvestigationDb.Open();
                var deleted = db.DeleteMatching(scope.ModelSha, scope.CorpusSha, scope.ImatrixSha, scope.ContextSize);
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    DbRowCountMatchingSelection = 0;
                    DbRowCountForCampaign = 0;
                    DbStatusLine = $"Deleted {deleted} rows. Campaign starts fresh on next Build.";
                });
            });
        }
        catch (Exception ex)
        {
            DbStatusLine = $"Invalidation failed: {ex.Message}";
        }
    }

    /// <summary>Resolved campaign-identity tuple + selection. Internal helper for the DB controls.</summary>
    private sealed record DbScope(
        string ModelSha,
        string CorpusSha,
        string ImatrixSha,
        int ContextSize,
        IReadOnlyList<string> Targets,
        IReadOnlyList<LlamaTensorType> Types);

    public override void ApplyActiveModel(string? path)
    {
        if (string.IsNullOrEmpty(SourceModelPath)
            && ResolveGgufFromActive(path) is { } resolved)
            SourceModelPath = resolved;
        if (string.IsNullOrEmpty(ImatrixPath)
            && ResolveImatrixForGguf(SourceModelPath) is { } imt)
            ImatrixPath = imt;
    }

    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(SourceModelPath);
    protected override bool HasImatrixSlot => true;
    protected override bool HasImatrixInputValue => !string.IsNullOrEmpty(ImatrixPath);
    protected override string? CurrentSourceGguf =>
        string.IsNullOrEmpty(SourceModelPath) ? ResolveGgufFromActive(Active?.Path) : SourceModelPath;

    partial void OnSourceModelPathChanged(string value)
    {
        RefreshArchitectureFromSource();
        NotifyRemediesChanged();
    }
    partial void OnImatrixPathChanged(string value) => NotifyRemediesChanged();

    partial void OnWorkingDirectoryChanged(string value)
    {
        // Persist the user's choice so the scratch dir survives app restarts.
        // Treat empty as "use default" (null in settings); avoids storing a
        // tombstone entry for the in-memory empty default.
        var newSetting = string.IsNullOrWhiteSpace(value) ? null : value;
        if (_settings.ProfileBuilderScratchDirectory != newSetting)
        {
            _settings.ProfileBuilderScratchDirectory = newSetting;
            try { _settings.Save(); } catch { /* best-effort persistence */ }
        }
    }

    /// <summary>
    /// One row in the progress grid — covers a single ablation target
    /// (a category or a tensor) across all candidate types. Cells are
    /// indexed positionally to match <see cref="ProgressColumnHeaders"/>.
    /// </summary>
    public sealed class ProgressRow
    {
        public string Label { get; }
        public ObservableCollection<ProgressCell?> Cells { get; }

        public ProgressRow(string label, int cellCount)
        {
            Label = label;
            Cells = new ObservableCollection<ProgressCell?>(new ProgressCell?[cellCount]);
        }
    }

    /// <summary>
    /// One cell in the progress grid. State + delta are observable so
    /// per-cell updates flow through the existing layout — the grid
    /// shape itself is fixed by <see cref="ProgressRow.Cells"/>'s
    /// allocated capacity.
    /// </summary>
    public sealed partial class ProgressCell : ObservableObject
    {
        public LlamaTensorType Type { get; }

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(Glyph))]
        [NotifyPropertyChangedFor(nameof(IsActive))]
        [NotifyPropertyChangedFor(nameof(IsDone))]
        private LlamaSensitivityProfileBuilder.CellState _state = LlamaSensitivityProfileBuilder.CellState.Pending;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(DeltaText))]
        private double? _delta;

        internal ProfileBuilderViewModel? Owner { get; set; }

        public ProgressCell(LlamaTensorType type) { Type = type; }

        public string Glyph => State switch
        {
            LlamaSensitivityProfileBuilder.CellState.Pending     => "·",
            LlamaSensitivityProfileBuilder.CellState.Resumed     => "✓",
            LlamaSensitivityProfileBuilder.CellState.Quantizing  => "Q",
            LlamaSensitivityProfileBuilder.CellState.Scoring     => "S",
            LlamaSensitivityProfileBuilder.CellState.Done        => "✓",
            LlamaSensitivityProfileBuilder.CellState.Errored     => "✗",
            _                                                    => " ",
        };

        public bool IsActive => State is LlamaSensitivityProfileBuilder.CellState.Quantizing
                                        or LlamaSensitivityProfileBuilder.CellState.Scoring;
        public bool IsDone => State is LlamaSensitivityProfileBuilder.CellState.Done
                                     or LlamaSensitivityProfileBuilder.CellState.Resumed;

        public string DeltaText => Delta is double d ? d.ToString("F2") : "";
    }

    /// <summary>One row in the categories checkbox list.</summary>
    public sealed partial class CategoryItem : ObservableObject
    {
        public string Name { get; }
        [ObservableProperty]
        private bool _isSelected;

        // Set by the owning VM after construction so toggle events bubble
        // back as cost-estimate refreshes. Leaving it nullable avoids a
        // mandatory ctor parameter that'd complicate XAML usage.
        internal ProfileBuilderViewModel? Owner { get; set; }

        public CategoryItem(string name, bool isSelected)
        {
            Name = name;
            _isSelected = isSelected;
        }

        partial void OnIsSelectedChanged(bool value) =>
            Owner?.OnPropertyChanged(nameof(CostEstimateText));
    }
}
