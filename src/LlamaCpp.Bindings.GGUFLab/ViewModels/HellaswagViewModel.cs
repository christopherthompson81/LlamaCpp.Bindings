using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the HellaSwag page: pick a model + dataset file (or fetch
/// the standard one), set a task subset size, run scoring, report
/// the canonical <c>acc_norm</c> accuracy metric.
/// </summary>
public sealed partial class HellaswagViewModel : ToolPageViewModel
{
    public override string Title => "HellaSwag";
    public override string Description =>
        "Multiple-choice common-sense reasoning benchmark — picks the most-likely candidate continuation per task.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty] private string _modelPath = string.Empty;
    [ObservableProperty] private string _datasetPath = string.Empty;

    [ObservableProperty] private int _maxTasks = 500;
    [ObservableProperty] private int _contextSize = 512;
    [ObservableProperty] private int _gpuLayerCount = -1;
    [ObservableProperty] private int _threadCount = -1;

    [ObservableProperty] private string _statusLine = "Idle.";
    [ObservableProperty] private string _resultText = string.Empty;
    [ObservableProperty] private double _progressFraction;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    public string LogText => _logBuilder.ToString();

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public HellaswagViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    [RelayCommand]
    private async Task LoadStandardDatasetAsync()
    {
        if (IsRunning) return;
        try
        {
            StatusLine = HellaswagDataset.IsCached()
                ? "Loading cached HellaSwag validation set…"
                : "Downloading HellaSwag validation set (~10 K tasks)…";
            var progress = new Progress<(long downloaded, long? total)>(p =>
            {
                if (p.total is long t && t > 0)
                {
                    int pct = (int)(100L * p.downloaded / t);
                    StatusLine = $"Downloading HellaSwag… {p.downloaded / 1024:N0} / {t / 1024:N0} KB ({pct}%)";
                }
            });
            DatasetPath = await HellaswagDataset.EnsureAsync(progress);
            StatusLine = $"Loaded standard HellaSwag dataset.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to load HellaSwag: {ex.Message}";
        }
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(ModelPath) || !File.Exists(ModelPath))
        {
            StatusLine = "Pick a model GGUF first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(DatasetPath) || !File.Exists(DatasetPath))
        {
            StatusLine = "Pick a HellaSwag dataset file (or click Load standard…).";
            return;
        }

        _logBuilder.Clear();
        OnPropertyChanged(nameof(LogText));
        ResultText = string.Empty;
        ProgressFraction = 0;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        StatusLine = "Loading model + parsing dataset…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var result = await Task.Run(async () =>
            {
                var allTasks = LlamaHellaswag.ParseUpstreamFile(DatasetPath);
                IReadOnlyList<LlamaHellaswagTask> tasks = MaxTasks > 0 && allTasks.Count > MaxTasks
                    ? new ArraySegment<LlamaHellaswagTask>((LlamaHellaswagTask[])allTasks, 0, MaxTasks)
                    : allTasks;

                using var model = new LlamaModel(ModelPath, new LlamaModelParameters
                {
                    GpuLayerCount = GpuLayerCount,
                    UseMmap = true,
                });

                StatusLine = $"Scoring {tasks.Count:N0} tasks…";
                var progress = new Progress<LlamaHellaswagProgress>(p =>
                {
                    ProgressFraction = p.TaskCount > 0
                        ? (double)p.TaskIndex / p.TaskCount
                        : 0;
                    StatusLine = $"Task {p.TaskIndex}/{p.TaskCount} — running acc_norm = {p.RunningAccuracy * 100:F2}%";
                });

                return await LlamaHellaswag.ComputeAsync(model, tasks,
                    new LlamaHellaswagOptions
                    {
                        ContextSize = ContextSize,
                        ThreadCount = ThreadCount,
                    },
                    progress,
                    _cts.Token);
            }, _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";
            ResultText =
                $"acc_norm:           {result.AccuracyNorm * 100:F2}%   ({result.CorrectNorm:N0} / {result.TaskCount:N0})\n" +
                $"acc_raw:            {result.AccuracyRaw  * 100:F2}%   ({result.CorrectRaw:N0} / {result.TaskCount:N0})\n" +
                $"\n" +
                $"Tasks scored:       {result.TaskCount:N0}\n" +
                $"Effective context:  {result.ContextSize}\n" +
                $"Elapsed:            {result.Elapsed.TotalSeconds:F2}s\n" +
                $"\n" +
                $"acc_norm is the standard metric (length-normalized log-likelihood argmax).\n" +
                $"acc_raw shown for comparison; biased toward shorter endings.";
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed: {ex.Message}";
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
        StatusLine = "Cancellation requested.";
    }

    public override void ApplyActiveModel(string? path)
    {
        if (!string.IsNullOrEmpty(path) && string.IsNullOrEmpty(ModelPath)) ModelPath = path;
    }
}
