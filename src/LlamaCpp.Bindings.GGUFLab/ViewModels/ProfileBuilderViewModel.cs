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
    private string _corpusPath = string.Empty;

    [ObservableProperty]
    private string _imatrixPath = string.Empty;

    [ObservableProperty]
    private string _workingDirectory = string.Empty;

    [ObservableProperty]
    private string _outputProfilePath = string.Empty;

    [ObservableProperty]
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

    /// <summary>Tensor count auto-derived for per-layer mode (read-only).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CostEstimateText))]
    private int _perLayerTensorCount;

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
                targetCount = PerLayerTensorCount;
                targetLabel = "tensors";
                if (targetCount == 0)
                    return "Pick a source model to derive per-layer tensor count.";
            }
            int total = 1 + targetCount * typeCount;    // +1 for baseline
            return $"≈ {total} PPL runs ({targetCount} {targetLabel} × {typeCount} types + 1 baseline). " +
                   "Already-measured cells in the DB are skipped.";
        }
    }

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public ProfileBuilderViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
        Categories.CollectionChanged += (_, _) => OnPropertyChanged(nameof(CostEstimateText));
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

            if (Mode == CampaignMode.PerCategory)
            {
                var profile = await LlamaSensitivityProfileBuilder.BuildAsync(
                    SourceModelPath, CorpusPath, opts, progress, _cts.Token);

                profile.SaveToJson(OutputProfilePath);
                var elapsed = DateTime.Now - startedAt;
                StatusLine =
                    $"Wrote {OutputProfilePath} in {elapsed.TotalMinutes:F1} min " +
                    $"(arch={profile.ArchitectureId}, layers={profile.LayerCount}, F16 PPL={profile.F16BaselinePerplexity:F3}).";
            }
            else
            {
                var newCount = await LlamaSensitivityProfileBuilder.BuildPerLayerAsync(
                    SourceModelPath, CorpusPath, targetTensors: null, opts, progress, _cts.Token);
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
