using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the Profile Builder page: run the per-category PPL ablation
/// campaign that produces a <see cref="LlamaSensitivityProfile"/>. This
/// is hours of compute on a typical 1.7B-class model — the Adaptive
/// Quantization page consumes the resulting profile.json.
/// </summary>
public sealed partial class ProfileBuilderViewModel : ToolPageViewModel
{
    public override string Title => "Profile Builder";
    public override string Description =>
        "Build a sensitivity profile for a model: ablate each tensor category at each candidate type, measure ΔPPL. Long-running.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _sourceModelPath = string.Empty;

    [ObservableProperty]
    private string _corpusPath = string.Empty;

    [ObservableProperty]
    private string _imatrixPath = string.Empty;

    [ObservableProperty]
    private string _workingDirectory = string.Empty;

    [ObservableProperty]
    private string _outputProfilePath = string.Empty;

    /// <summary>
    /// PPL context size — defaults to 512 to match published wiki.test
    /// numbers (matches GGUFLab's standalone Perplexity tool).
    /// </summary>
    [ObservableProperty]
    private int _contextSize = 512;

    [ObservableProperty]
    private double _kneeDeltaPplThreshold = 5.0;

    /// <summary>
    /// 0 → auto (size-aware via <see cref="LlamaPerplexity.RecommendConcurrency"/>).
    /// </summary>
    [ObservableProperty]
    private int _maxConcurrent;

    [ObservableProperty]
    private double _availableVramGb = 24.0;

    [ObservableProperty]
    private bool _cleanupWorkingDirectory = true;

    /// <summary>Comma-separated list. Default = the seven canonical weight categories.</summary>
    [ObservableProperty]
    private string _categoriesText = "attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight, ffn_up, ffn_gate, ffn_down";

    [ObservableProperty]
    private bool _useQ2K = true;
    [ObservableProperty]
    private bool _useQ3K;
    [ObservableProperty]
    private bool _useQ4K = true;
    [ObservableProperty]
    private bool _useQ5K;
    [ObservableProperty]
    private bool _useQ6K = true;
    [ObservableProperty]
    private bool _useQ8_0;
    [ObservableProperty]
    private bool _useIQ4XS;

    [ObservableProperty]
    private string _statusLine = "Idle. Pick a source model and corpus.";

    [ObservableProperty]
    private string _stageText = string.Empty;

    [ObservableProperty]
    private double _progressFraction;

    [ObservableProperty]
    private string _progressLabel = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    public string LogText => _logBuilder.ToString();

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public ProfileBuilderViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
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
        var categories = ResolveCategories();
        if (categories.Count == 0)
        {
            StatusLine = "Pick at least one category.";
            return;
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
        StatusLine = "Building profile…";

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
                Categories              = categories,
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
            });
            var profile = await LlamaSensitivityProfileBuilder.BuildAsync(
                SourceModelPath, CorpusPath, opts, progress, _cts.Token);

            profile.SaveToJson(OutputProfilePath);
            var elapsed = DateTime.Now - startedAt;
            StatusLine =
                $"Wrote {OutputProfilePath} in {elapsed.TotalMinutes:F1} min " +
                $"(arch={profile.ArchitectureId}, layers={profile.LayerCount}, F16 PPL={profile.F16BaselinePerplexity:F3}).";
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled. (Checkpoint preserved — re-run to resume.)";
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
        var picks = new List<LlamaTensorType>();
        if (UseQ2K)    picks.Add(LlamaTensorType.Q2_K);
        if (UseQ3K)    picks.Add(LlamaTensorType.Q3_K);
        if (UseQ4K)    picks.Add(LlamaTensorType.Q4_K);
        if (UseIQ4XS)  picks.Add(LlamaTensorType.IQ4_XS);
        if (UseQ5K)    picks.Add(LlamaTensorType.Q5_K);
        if (UseQ6K)    picks.Add(LlamaTensorType.Q6_K);
        if (UseQ8_0)   picks.Add(LlamaTensorType.Q8_0);
        return picks;
    }

    private IReadOnlyList<string> ResolveCategories()
    {
        if (string.IsNullOrWhiteSpace(CategoriesText)) return Array.Empty<string>();
        return CategoriesText
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();
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

    partial void OnSourceModelPathChanged(string value) => NotifyRemediesChanged();
    partial void OnImatrixPathChanged(string value) => NotifyRemediesChanged();
}
