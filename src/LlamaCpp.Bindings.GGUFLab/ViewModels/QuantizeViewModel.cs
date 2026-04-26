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
/// Drives the Quantize page: input/output paths, ftype selection,
/// thread/options form, run/cancel buttons, and a streaming log pane fed by
/// llama.cpp's native log sink (installed once at app startup, see
/// <see cref="App"/>).
/// </summary>
public sealed partial class QuantizeViewModel : ToolPageViewModel
{
    public override string Title => "Quantize";
    public override string Description =>
        "Re-encode a GGUF model to a smaller quant. Wraps llama_model_quantize directly — no external CLI.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _inputPath = string.Empty;

    [ObservableProperty]
    private string _outputPath = string.Empty;

    [ObservableProperty]
    private LlamaFileType _selectedFileType = LlamaFileType.Q4_K_M;

    /// <summary>Items source for the ftype combobox.</summary>
    public IReadOnlyList<LlamaFileType> AvailableFileTypes { get; } =
        Enum.GetValues<LlamaFileType>()
            .Where(f => f != LlamaFileType.Guessed)
            .OrderBy(f => f.ToString(), StringComparer.Ordinal)
            .ToArray();

    [ObservableProperty]
    private int _threadCount;

    [ObservableProperty]
    private bool _allowRequantize;

    [ObservableProperty]
    private bool _quantizeOutputTensor = true;

    [ObservableProperty]
    private bool _onlyCopy;

    [ObservableProperty]
    private bool _pure;

    [ObservableProperty]
    private bool _keepSplit;

    [ObservableProperty]
    private bool _dryRun;

    [ObservableProperty]
    private string _statusLine = "Idle.";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    /// <summary>Streaming text shown in the log pane.</summary>
    public string LogText
    {
        get => _logBuilder.ToString();
    }

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public QuantizeViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(InputPath))
        {
            StatusLine = "Pick an input GGUF first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            // Default the output to <input>.<ftype>.gguf next to the source.
            try
            {
                var dir = Path.GetDirectoryName(InputPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(InputPath);
                OutputPath = Path.Combine(dir, $"{stem}.{SelectedFileType}.gguf");
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
        StatusLine = $"Running… ({SelectedFileType})";

        // Subscribe to native logs for this run only — they're noisy and we
        // don't want stale lines bleeding into the next run.
        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            // Property change must marshal back to the UI thread; the bus
            // already does that for its subscribers.
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var parameters = new LlamaQuantizationParameters
            {
                FileType             = SelectedFileType,
                ThreadCount          = ThreadCount,
                AllowRequantize      = AllowRequantize,
                QuantizeOutputTensor = QuantizeOutputTensor,
                OnlyCopy             = OnlyCopy,
                Pure                 = Pure,
                KeepSplit            = KeepSplit,
                DryRun               = DryRun,
            };

            await LlamaQuantizer.QuantizeAsync(
                InputPath, OutputPath, parameters, _cts.Token);

            var elapsed = DateTime.Now - startedAt;
            StatusLine = DryRun
                ? $"Dry-run complete in {elapsed.TotalSeconds:F1}s."
                : $"Wrote {OutputPath} in {elapsed.TotalSeconds:F1}s.";
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
        // Pre-flight cancellation only — once the native llama_model_quantize
        // is running, it ignores cancellation. The status line updates so the
        // user knows what happened.
        _cts?.Cancel();
        StatusLine = IsRunning
            ? "Cancellation requested — will take effect at the next checkpoint."
            : StatusLine;
    }
}
