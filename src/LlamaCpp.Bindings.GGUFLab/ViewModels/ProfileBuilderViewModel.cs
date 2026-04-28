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

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
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

    public string LogText => _logBuilder.ToString();

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

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public ProfileBuilderViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
        Categories.CollectionChanged += (_, _) => OnPropertyChanged(nameof(CostEstimateText));
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

        _logBuilder.Clear();
        OnPropertyChanged(nameof(LogText));
        ProgressFraction = 0;
        ProgressLabel = string.Empty;
        StageText = "starting…";
        _cts = new CancellationTokenSource();
        IsRunning = true;
        var startedAt = DateTime.Now;
        StatusLine = Mode == CampaignMode.PerCategory
            ? "Building per-category profile…"
            : "Building per-layer profile…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

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
                if (p.TotalJobs > 0)
                    ProgressFraction = (double)p.CompletedJobs / p.TotalJobs;
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
            _logBuilder.AppendLine($"[error] {ex}");
            OnPropertyChanged(nameof(LogText));
        }
        finally
        {
            unsubscribe();
            IsRunning = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

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
