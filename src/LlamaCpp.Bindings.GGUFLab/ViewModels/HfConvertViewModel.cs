using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;
using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the HuggingFace → GGUF page: pick an HF model directory,
/// pick output type (F32/F16/BF16), run the converter. V1 supports
/// architectures registered in <see cref="LlamaHfConverter"/>'s
/// embedded definition library — currently Qwen3 dense.
/// </summary>
public sealed partial class HfConvertViewModel : ToolPageViewModel
{
    public override string Title => "Huggingface to GGUF";
    public override string Description =>
        "Convert a HuggingFace model directory to a GGUF file. Pure C# — no Python, no external CLI.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _hfDirectory = string.Empty;

    [ObservableProperty]
    private string _outputPath = string.Empty;

    [ObservableProperty]
    private LlamaHfConvertOutputType _outputType = LlamaHfConvertOutputType.F16;

    public IReadOnlyList<LlamaHfConvertOutputType> AvailableOutputTypes { get; } =
        new[]
        {
            LlamaHfConvertOutputType.F16,
            LlamaHfConvertOutputType.BF16,
            LlamaHfConvertOutputType.F32,
        };

    /// <summary>Architecture names this build's converter supports — shown in the form for context.</summary>
    public string SupportedArchitecturesText =>
        string.Join(", ", LlamaHfConverter.AvailableArchitectures());

    [ObservableProperty]
    private string _statusLine = "Pick an HF model directory to begin.";

    [ObservableProperty]
    private string _resultText = string.Empty;

    [ObservableProperty]
    private double _progressFraction;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    public string LogText => _log.Text;

    /// <summary>Throttled log surface — same pattern as the other long-running pages.</summary>
    private readonly ThrottledLogBuffer _log = new();
    private CancellationTokenSource? _cts;

    public HfConvertViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
        _log.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(ThrottledLogBuffer.Text))
                OnPropertyChanged(nameof(LogText));
        };
    }

    public override void ApplyActiveModel(string? path)
    {
        // The active model can be a directory (a replicated safetensors
        // model) — that's exactly what this tool wants. Ignore .gguf
        // files; converting from GGUF isn't this tool's job.
        if (string.IsNullOrEmpty(path)) return;
        if (!string.IsNullOrEmpty(HfDirectory)) return;
        if (Directory.Exists(path)) HfDirectory = path;
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(HfDirectory) || !Directory.Exists(HfDirectory))
        {
            StatusLine = "Pick an HF model directory first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            try
            {
                var stem = new DirectoryInfo(HfDirectory).Name;
                OutputPath = Path.Combine(HfDirectory, $"{stem}.{OutputType}.gguf");
            }
            catch
            {
                StatusLine = "Pick an output path.";
                return;
            }
        }

        _log.Clear();
        ResultText = string.Empty;
        ProgressFraction = 0;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        StatusLine = "Reading HF model…";

        var unsubscribe = _logBus.Subscribe(line => _log.Append(line));

        try
        {
            var progress = new Progress<LlamaHfConvertProgress>(p =>
            {
                ProgressFraction = p.TensorCount > 0
                    ? (double)p.TensorIndex / p.TensorCount
                    : 0;
                StatusLine = $"Converting tensor {p.TensorIndex}/{p.TensorCount} — {p.CurrentTensorName}";
            });

            var result = await LlamaHfConverter.ConvertAsync(
                HfDirectory, OutputPath, OutputType, progress, _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";
            ResultText =
                $"Output:               {result.OutputPath}\n" +
                $"Architecture:         {result.Architecture}\n" +
                $"Output type:          {result.OutputType}\n" +
                $"Tensors:              {result.TensorCount:N0}\n" +
                $"Output bytes:         {result.OutputBytes:N0}\n" +
                $"Elapsed:              {result.Elapsed.TotalSeconds:F2}s";

            // The active model is the model FOLDER, not any single file
            // inside it — that's what lets the user select once and walk
            // through Convert → Imatrix → Quantize → Adaptive without
            // re-selecting. Pin the folder; downstream tools resolve the
            // appropriate format (the new F16 here) on activation.
            var modelDir = Path.GetDirectoryName(result.OutputPath);
            if (!string.IsNullOrEmpty(modelDir)) Active?.Set(modelDir);
        }
        catch (OperationCanceledException)
        {
            StatusLine = "Cancelled.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed: {ex.Message}";
            _log.Append($"[error] {ex}");
        }
        finally
        {
            unsubscribe();
            _log.Stop();
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
}
