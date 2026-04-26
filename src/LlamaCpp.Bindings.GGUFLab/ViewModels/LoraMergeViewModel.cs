using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;
using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the LoRA Merge page: pick a base model + LoRA adapter +
/// scale + output GGUF, run <see cref="LlamaLoraMerge"/>, report the
/// merged-vs-copied tensor counts.
/// </summary>
public sealed partial class LoraMergeViewModel : ToolPageViewModel
{
    public override string Title => "LoRA Merge";
    public override string Description =>
        "Apply a LoRA adapter to a base model and write a single merged GGUF. Pure C# F32 matmul, F16 output by default.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty] private string _basePath = string.Empty;
    [ObservableProperty] private string _adapterPath = string.Empty;
    [ObservableProperty] private float _adapterScale = 1.0f;
    [ObservableProperty] private string _outputPath = string.Empty;

    [ObservableProperty] private LlamaHfConvertOutputType _outputType = LlamaHfConvertOutputType.F16;

    public IReadOnlyList<LlamaHfConvertOutputType> AvailableOutputTypes { get; } =
        new[]
        {
            LlamaHfConvertOutputType.F16,
            LlamaHfConvertOutputType.BF16,
            LlamaHfConvertOutputType.F32,
        };

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

    public LoraMergeViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(BasePath) || !File.Exists(BasePath))
        {
            StatusLine = "Pick a base model GGUF first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(AdapterPath) || !File.Exists(AdapterPath))
        {
            StatusLine = "Pick a LoRA adapter GGUF first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            try
            {
                var dir = Path.GetDirectoryName(BasePath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(BasePath);
                OutputPath = Path.Combine(dir, $"{stem}.merged.gguf");
            }
            catch
            {
                StatusLine = "Pick an output path.";
                return;
            }
        }

        _logBuilder.Clear();
        OnPropertyChanged(nameof(LogText));
        ResultText = string.Empty;
        ProgressFraction = 0;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        StatusLine = "Merging…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var progress = new Progress<LlamaLoraMergeProgress>(p =>
            {
                ProgressFraction = p.TensorCount > 0
                    ? (double)p.TensorIndex / p.TensorCount
                    : 0;
                StatusLine = $"Tensor {p.TensorIndex}/{p.TensorCount} — {(p.IsMerged ? "merge" : "copy")} {p.CurrentTensorName}";
            });

            var result = await LlamaLoraMerge.MergeAsync(
                BasePath,
                new[] { new LlamaLoraAdapterInput(AdapterPath, AdapterScale) },
                OutputPath,
                new LlamaLoraMergeOptions { OutputType = OutputType },
                progress,
                _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";
            ResultText =
                $"Output:               {OutputPath}\n" +
                $"Output type:          {result.OutputType}\n" +
                $"Tensors merged:       {result.TensorsMerged:N0}\n" +
                $"Tensors copied:       {result.TensorsCopied:N0}\n" +
                $"Tensors total:        {result.TensorsTotal:N0}\n" +
                $"Output bytes:         {result.OutputBytes:N0}\n" +
                $"Elapsed:              {result.Elapsed.TotalSeconds:F2}s";
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
        if (string.IsNullOrEmpty(BasePath)
            && ResolveGgufFromActive(path) is { } resolved)
            BasePath = resolved;
    }

    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(BasePath);

    partial void OnBasePathChanged(string value) => NotifyRemediesChanged();
}
