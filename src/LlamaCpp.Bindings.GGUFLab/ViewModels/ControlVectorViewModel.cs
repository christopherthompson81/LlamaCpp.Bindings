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
/// Drives the Control Vectors page: pick a model + paired
/// positive/negative prompts, train via <see cref="LlamaControlVectorTrainer"/>,
/// save the resulting GGUF for use with <see cref="LlamaContext.SetControlVector"/>.
/// </summary>
public sealed partial class ControlVectorViewModel : ToolPageViewModel
{
    public override string Title => "Control Vectors";
    public override string Description =>
        "Train per-layer steering vectors from contrastive prompt pairs. Pure C# eval-callback collection + PCA/mean reduction.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _modelPath = string.Empty;

    /// <summary>
    /// Positive prompts as raw text — one prompt per non-empty line.
    /// Matches the format expected by <c>llama-cvector-generator</c>.
    /// </summary>
    [ObservableProperty]
    private string _positivePromptsText = string.Empty;

    [ObservableProperty]
    private string _negativePromptsText = string.Empty;

    [ObservableProperty]
    private string _outputPath = string.Empty;

    [ObservableProperty]
    private LlamaControlVectorMethod _method = LlamaControlVectorMethod.Pca;

    public IReadOnlyList<LlamaControlVectorMethod> AvailableMethods { get; } =
        Enum.GetValues<LlamaControlVectorMethod>();

    [ObservableProperty]
    private int _pcaIterations = 1000;

    [ObservableProperty]
    private int _gpuLayerCount = -1;

    [ObservableProperty]
    // Physical-core default (logical/2). See ImatrixViewModel for rationale.
    private int _threadCount = Math.Max(1, Environment.ProcessorCount / 2);

    [ObservableProperty]
    private string _statusLine = "Idle.";

    [ObservableProperty]
    private string _resultText = string.Empty;

    [ObservableProperty]
    private double _progressFraction;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    public string LogText => _logBuilder.ToString();

    private readonly System.Text.StringBuilder _logBuilder = new();
    private CancellationTokenSource? _cts;

    public ControlVectorViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    /// <summary>Read prompts file contents into the corresponding text box. Called from the view's code-behind.</summary>
    public async Task LoadPositiveFromFileAsync(string path)
    {
        try
        {
            PositivePromptsText = await File.ReadAllTextAsync(path);
            StatusLine = $"Loaded positive prompts ({SplitPrompts(PositivePromptsText).Count} lines).";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to load positive prompts: {ex.Message}";
        }
    }

    public async Task LoadNegativeFromFileAsync(string path)
    {
        try
        {
            NegativePromptsText = await File.ReadAllTextAsync(path);
            StatusLine = $"Loaded negative prompts ({SplitPrompts(NegativePromptsText).Count} lines).";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to load negative prompts: {ex.Message}";
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

        var positive = SplitPrompts(PositivePromptsText);
        var negative = SplitPrompts(NegativePromptsText);
        if (positive.Count == 0)
        {
            StatusLine = "Add at least one positive prompt (one per line).";
            return;
        }
        if (positive.Count != negative.Count)
        {
            StatusLine = $"Positive ({positive.Count}) and negative ({negative.Count}) prompt counts must match.";
            return;
        }

        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            try
            {
                var dir = Path.GetDirectoryName(ModelPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(ModelPath);
                OutputPath = Path.Combine(dir, $"{stem}.cvector.gguf");
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
        StatusLine = "Loading model…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var result = await Task.Run(async () =>
            {
                using var model = new LlamaModel(ModelPath, new LlamaModelParameters
                {
                    GpuLayerCount = GpuLayerCount,
                    UseMmap = true,
                });

                StatusLine = "Training…";

                var progress = new Progress<LlamaControlVectorProgress>(p =>
                {
                    ProgressFraction = p.PromptPairCount > 0
                        ? (double)p.PromptPairIndex / p.PromptPairCount
                        : 0;
                    StatusLine = $"Pair {p.PromptPairIndex}/{p.PromptPairCount} — {p.Phase}";
                });

                return await LlamaControlVectorTrainer.ComputeAsync(
                    model,
                    positive,
                    negative,
                    OutputPath,
                    new LlamaControlVectorOptions
                    {
                        Method        = Method,
                        PcaIterations = PcaIterations,
                        ThreadCount   = ThreadCount,
                    },
                    progress,
                    _cts.Token);
            }, _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";
            ResultText =
                $"Output:               {OutputPath}\n" +
                $"Method:               {result.Method}\n" +
                $"Layers:               {result.LayerCount:N0}\n" +
                $"Embedding size:       {result.EmbeddingSize:N0}\n" +
                $"Prompt pairs:         {result.PromptPairCount:N0}\n" +
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

    /// <summary>Split a multi-line prompt blob into one prompt per non-empty line.</summary>
    private static IReadOnlyList<string> SplitPrompts(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();
        return text.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(l => l.TrimEnd('\r').Trim())
            .Where(l => l.Length > 0)
            .ToArray();
    }

    public override void ApplyActiveModel(string? path)
    {
        if (string.IsNullOrEmpty(ModelPath)
            && ResolveGgufFromActive(path) is { } resolved)
            ModelPath = resolved;
    }

    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(ModelPath);

    partial void OnModelPathChanged(string value) => NotifyRemediesChanged();
}
