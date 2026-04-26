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
/// Drives the Adaptive Quantization page: per-tensor sensitivity sweep →
/// threshold-driven recipe → apply via <c>tt_overrides</c>. The sweep is
/// the expensive step (one quantize+dequantize per (tensor, candidate));
/// the recipe is rebuilt for free as the user moves the threshold slider.
/// </summary>
public sealed partial class AdaptiveQuantizeViewModel : ToolPageViewModel
{
    public override string Title => "Adaptive Quantization";
    public override string Description =>
        "Score each tensor's quantization sensitivity, then pin the smallest type that meets a quality threshold.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _inputPath = string.Empty;

    [ObservableProperty]
    private string _outputPath = string.Empty;

    [ObservableProperty]
    private string _imatrixPath = string.Empty;

    /// <summary>Fallback ftype the production quantizer uses for tensors the recipe doesn't cover.</summary>
    [ObservableProperty]
    private LlamaFileType _baseFileType = LlamaFileType.Q4_K_M;

    public IReadOnlyList<LlamaFileType> AvailableFileTypes { get; } =
        Enum.GetValues<LlamaFileType>()
            .Where(f => f != LlamaFileType.Guessed)
            .OrderBy(f => f.ToString(), StringComparer.Ordinal)
            .ToArray();

    /// <summary>
    /// Relative-MSE threshold τ. Smaller = stricter quality, larger
    /// recipes; bigger = more aggressive compression. Slider range
    /// covers the most useful band; users can type values outside it.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(AverageBitsPerElementText))]
    [NotifyPropertyChangedFor(nameof(ExceededCount))]
    private double _threshold = 0.05;

    [ObservableProperty]
    private int _sweepProgressCurrent;

    [ObservableProperty]
    private int _sweepProgressTotal;

    [ObservableProperty]
    private string _statusLine = "Idle. Pick an input GGUF and run the sensitivity sweep.";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    [NotifyPropertyChangedFor(nameof(HasScores))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasScores))]
    [NotifyPropertyChangedFor(nameof(AverageBitsPerElementText))]
    [NotifyPropertyChangedFor(nameof(ExceededCount))]
    private LlamaQuantSensitivityResult? _scores;

    public bool HasScores => Scores is not null && Scores.Scores.Count > 0;

    /// <summary>Recipe rebuilt every time threshold or scores change. Drives the preview grid.</summary>
    public ObservableCollection<RecipeRow> RecipeRows { get; } = new();

    public string AverageBitsPerElementText
    {
        get
        {
            if (Scores is null) return "";
            var recipe = BuildRecipeOrNull();
            if (recipe is null || double.IsNaN(recipe.AverageBitsPerElement)) return "";
            return $"avg {recipe.AverageBitsPerElement:F2} bpw";
        }
    }

    public int ExceededCount => RecipeRows.Count(r => r.ExceededThreshold);

    public string LogText => _logBuilder.ToString();

    /// <summary>
    /// One chip per candidate type for the tensor currently being scored.
    /// Drives the lightboard so the user always sees what's in flight,
    /// what's done with its score, and what's pending — even when a
    /// single tensor takes minutes (token_embd in IQ-quants etc.).
    /// </summary>
    public ObservableCollection<CandidateChip> CandidateChips { get; } = new();

    [ObservableProperty]
    private string _currentTensorName = string.Empty;

    [ObservableProperty]
    private string _currentSubStatus = string.Empty;

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public AdaptiveQuantizeViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    partial void OnThresholdChanged(double value) => RebuildRecipe();
    partial void OnScoresChanged(LlamaQuantSensitivityResult? value) => RebuildRecipe();

    [RelayCommand]
    private async Task MeasureAsync()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(InputPath))
        {
            StatusLine = "Pick an input GGUF first.";
            return;
        }
        if (!File.Exists(InputPath))
        {
            StatusLine = $"Input file not found: {InputPath}";
            return;
        }

        _logBuilder.Clear();
        OnPropertyChanged(nameof(LogText));
        SweepProgressCurrent = 0;
        SweepProgressTotal = 0;
        _cts = new CancellationTokenSource();
        IsRunning = true;
        var startedAt = DateTime.Now;
        StatusLine = "Running sensitivity sweep…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var options = new LlamaQuantSensitivityOptions
            {
                ImatrixPath = string.IsNullOrWhiteSpace(ImatrixPath) ? null : ImatrixPath,
            };
            var candidateList = options.CandidateTypes ?? LlamaQuantSensitivity.DefaultCandidateTypes;
            CandidateChips.Clear();
            CurrentTensorName = string.Empty;
            CurrentSubStatus = string.Empty;
            var progress = new Progress<LlamaQuantSensitivityProgress>(p =>
            {
                SweepProgressCurrent = p.TensorIndex;
                SweepProgressTotal = p.TensorCount;
                StatusLine = $"Sweep {p.TensorIndex}/{p.TensorCount} — {p.CurrentTensorName}";
                ApplyProgressToLightboard(p, candidateList);
            });
            var result = await LlamaQuantSensitivity.MeasureAsync(
                InputPath, options, progress, _cts.Token);

            Scores = result;
            var elapsed = DateTime.Now - startedAt;
            StatusLine = $"Swept {result.Scores.Select(s => s.TensorName).Distinct().Count()} tensors " +
                         $"× {result.CandidateTypes.Count} candidates in {elapsed.TotalSeconds:F1}s.";
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Sweep failed: {ex.Message}";
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

    [RelayCommand]
    private async Task ApplyAsync()
    {
        if (IsRunning) return;
        var recipe = BuildRecipeOrNull();
        if (recipe is null || recipe.Entries.Count == 0)
        {
            StatusLine = "No recipe to apply — run the sweep first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(InputPath) || !File.Exists(InputPath))
        {
            StatusLine = "Pick a valid input GGUF.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            try
            {
                var dir = Path.GetDirectoryName(InputPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(InputPath);
                OutputPath = Path.Combine(dir, $"{stem}.adaptive-{Threshold:F3}.gguf");
            }
            catch
            {
                StatusLine = "Pick an output path.";
                return;
            }
        }

        _logBuilder.Clear();
        OnPropertyChanged(nameof(LogText));
        _cts = new CancellationTokenSource();
        IsRunning = true;
        var startedAt = DateTime.Now;
        StatusLine = "Applying recipe…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var parameters = new LlamaQuantizationParameters
            {
                FileType            = BaseFileType,
                AllowRequantize     = true,    // sources are often already quantized
                Pure                = false,   // tt_overrides are gated behind !Pure
                TensorTypeOverrides = recipe.ToTtOverrides(),
            };
            await LlamaQuantizer.QuantizeAsync(InputPath, OutputPath, parameters, _cts.Token);

            var elapsed = DateTime.Now - startedAt;
            StatusLine = $"Wrote {OutputPath} in {elapsed.TotalSeconds:F1}s " +
                         $"({recipe.Entries.Count} pinned tensors).";
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Apply failed: {ex.Message}";
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

    /// <summary>Save the raw sensitivity table to JSON. Reusable across threshold sweeps.</summary>
    public void SaveScoresJson(string path)
    {
        if (Scores is null)
        {
            StatusLine = "No scores to save.";
            return;
        }
        try
        {
            LlamaQuantSensitivity.SaveToJson(Scores, path);
            StatusLine = $"Saved score table to {path}.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Save failed: {ex.Message}";
        }
    }

    public void LoadScoresJson(string path)
    {
        try
        {
            Scores = LlamaQuantSensitivity.LoadFromJson(path);
            StatusLine = $"Loaded {Scores.Scores.Count} score rows from {Path.GetFileName(path)}.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Load failed: {ex.Message}";
        }
    }

    public void SaveRecipeJson(string path)
    {
        var recipe = BuildRecipeOrNull();
        if (recipe is null)
        {
            StatusLine = "No recipe to save.";
            return;
        }
        try
        {
            LlamaQuantRecipe.SaveToJson(recipe, path);
            StatusLine = $"Saved recipe to {path}.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Save failed: {ex.Message}";
        }
    }

    private LlamaQuantRecipe? BuildRecipeOrNull()
    {
        if (Scores is null || Scores.Scores.Count == 0) return null;
        if (!(Threshold > 0)) return null;
        try
        {
            return LlamaQuantRecipe.Build(Scores, Threshold, sourceScoreTablePath: null);
        }
        catch
        {
            return null;
        }
    }

    private void RebuildRecipe()
    {
        RecipeRows.Clear();
        var recipe = BuildRecipeOrNull();
        if (recipe is null)
        {
            OnPropertyChanged(nameof(AverageBitsPerElementText));
            OnPropertyChanged(nameof(ExceededCount));
            return;
        }
        foreach (var e in recipe.Entries)
        {
            RecipeRows.Add(new RecipeRow(
                e.TensorName,
                e.ChosenType.ToString(),
                e.BitsPerElement,
                e.RelativeMse,
                e.ExceededThreshold));
        }
        OnPropertyChanged(nameof(AverageBitsPerElementText));
        OnPropertyChanged(nameof(ExceededCount));
    }

    private void ApplyProgressToLightboard(
        LlamaQuantSensitivityProgress p,
        IReadOnlyList<LlamaTensorType> candidates)
    {
        // New tensor → reset the chip strip so the user sees fresh
        // pending entries instead of the previous tensor's results.
        if (p.Phase == LlamaQuantSensitivityPhase.Tensor)
        {
            CurrentTensorName = p.CurrentTensorName;
            CurrentSubStatus  = "starting…";
            CandidateChips.Clear();
            for (int i = 0; i < candidates.Count; i++)
                CandidateChips.Add(new CandidateChip(i, candidates[i].ToString(), CandidateChipState.Pending, null));
            return;
        }

        if (p.Phase == LlamaQuantSensitivityPhase.SourceDequantize)
        {
            CurrentSubStatus = "reading + dequantizing source to F32…";
            return;
        }

        // Per-candidate phases. CandidateIndex maps directly to the chip
        // we initialized above.
        if (p.CandidateIndex < 0 || p.CandidateIndex >= CandidateChips.Count) return;

        var existing = CandidateChips[p.CandidateIndex];
        switch (p.Phase)
        {
            case LlamaQuantSensitivityPhase.Quantize:
                CandidateChips[p.CandidateIndex] = existing with { State = CandidateChipState.Quantizing };
                CurrentSubStatus = $"{existing.Type}: quantize";
                break;
            case LlamaQuantSensitivityPhase.Dequantize:
                CandidateChips[p.CandidateIndex] = existing with { State = CandidateChipState.Dequantizing };
                CurrentSubStatus = $"{existing.Type}: dequantize";
                break;
            case LlamaQuantSensitivityPhase.Score:
                CandidateChips[p.CandidateIndex] = existing with { State = CandidateChipState.Scoring };
                CurrentSubStatus = $"{existing.Type}: score";
                break;
            case LlamaQuantSensitivityPhase.CandidateDone:
                CandidateChips[p.CandidateIndex] = existing with
                {
                    State = CandidateChipState.Done,
                    RelativeMse = p.CandidateRelativeMse,
                };
                break;
        }
    }

    public override void ApplyActiveModel(string? path)
    {
        if (string.IsNullOrEmpty(InputPath)
            && ResolveGgufFromActive(path) is { } resolved)
            InputPath = resolved;
        // Back-fill the imatrix slot from the sidecar Imatrix produces
        // next to the source model. This is what makes "go build the
        // imatrix, then come back" work without a re-Browse.
        if (string.IsNullOrEmpty(ImatrixPath)
            && ResolveImatrixForGguf(InputPath) is { } imt)
            ImatrixPath = imt;
    }

    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(InputPath);
    protected override bool HasImatrixSlot => true;
    protected override bool HasImatrixInputValue => !string.IsNullOrEmpty(ImatrixPath);
    protected override string? CurrentSourceGguf =>
        string.IsNullOrEmpty(InputPath) ? ResolveGgufFromActive(Active?.Path) : InputPath;

    partial void OnInputPathChanged(string value) => NotifyRemediesChanged();
    partial void OnImatrixPathChanged(string value) => NotifyRemediesChanged();

    public sealed record RecipeRow(
        string TensorName,
        string ChosenType,
        double BitsPerElement,
        double RelativeMse,
        bool ExceededThreshold)
    {
        public string BitsPerElementText => double.IsNaN(BitsPerElement) ? "—" : $"{BitsPerElement:F2}";
        public string RelativeMseText => double.IsNaN(RelativeMse) ? "—" : RelativeMse.ToString("E2");
        public string FlagText => ExceededThreshold ? "⚠ over τ" : "";
    }

    public enum CandidateChipState { Pending, Quantizing, Dequantizing, Scoring, Done }

    /// <summary>
    /// One chip on the lightboard. The view styles itself off
    /// <see cref="State"/> so the user reads "what's running" at a
    /// glance without parsing the status line.
    /// </summary>
    public sealed record CandidateChip(
        int Index,
        string Type,
        CandidateChipState State,
        double? RelativeMse)
    {
        public string StateGlyph => State switch
        {
            CandidateChipState.Pending      => "·",
            CandidateChipState.Quantizing   => "Q…",
            CandidateChipState.Dequantizing => "D…",
            CandidateChipState.Scoring      => "S…",
            CandidateChipState.Done         => "✓",
            _ => "",
        };
        public string MseText => RelativeMse is double m ? m.ToString("E2") : "";
        public bool IsActive => State is CandidateChipState.Quantizing
                                      or CandidateChipState.Dequantizing
                                      or CandidateChipState.Scoring;
        public bool IsDone => State == CandidateChipState.Done;
    }
}
