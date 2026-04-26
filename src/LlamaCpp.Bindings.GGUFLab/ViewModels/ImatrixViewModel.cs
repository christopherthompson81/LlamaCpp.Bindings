using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the Importance Matrix page: pick a model + calibration corpus,
/// run <see cref="LlamaImatrix"/>, save the resulting GGUF for use by a
/// later quantize call.
/// </summary>
public sealed partial class ImatrixViewModel : ToolPageViewModel
{
    public override string Title => "Importance Matrix";
    public override string Description =>
        "Generate an imatrix GGUF from a calibration corpus — pure C# eval-callback collection, no external CLI.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _modelPath = string.Empty;

    [ObservableProperty]
    private string _corpusPath = string.Empty;

    [ObservableProperty]
    private string _corpusText = string.Empty;

    [ObservableProperty]
    private string _outputPath = string.Empty;

    [ObservableProperty]
    private int _contextSize = 512;

    [ObservableProperty]
    private bool _processOutput;

    [ObservableProperty]
    private int _gpuLayerCount = -1;

    [ObservableProperty]
    private int _threadCount = -1;

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

    public ImatrixViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    /// <summary>
    /// Read <paramref name="path"/> into <see cref="CorpusText"/> and
    /// remember the path. Called from the view's code-behind once a file
    /// has been picked.
    /// </summary>
    public async Task SetCorpusFromFileAsync(string path)
    {
        try
        {
            CorpusPath = path;
            CorpusText = await File.ReadAllTextAsync(path);
            StatusLine = $"Loaded {Path.GetFileName(path)} ({CorpusText.Length:N0} chars).";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to load corpus: {ex.Message}";
        }
    }

    /// <summary>Fetch (or read from cache) the wikitext-2 raw test set as a calibration corpus.</summary>
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
            await SetCorpusFromFileAsync(path);
            StatusLine = $"Loaded wikitext-2 test ({CorpusText.Length:N0} chars). " +
                         $"Common practice: use the full file with n_ctx=512 for a calibration matrix.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to load wikitext-2: {ex.Message}";
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
        if (string.IsNullOrWhiteSpace(CorpusText))
        {
            StatusLine = "Pick or paste a calibration corpus first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputPath))
        {
            // Default to <model>.imatrix.gguf next to the model.
            try
            {
                var dir = Path.GetDirectoryName(ModelPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(ModelPath);
                OutputPath = Path.Combine(dir, $"{stem}.imatrix.gguf");
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

                StatusLine = "Collecting…";

                var progress = new Progress<LlamaImatrixProgress>(p =>
                {
                    ProgressFraction = p.ChunkCount > 0
                        ? (double)p.ChunkIndex / p.ChunkCount
                        : 0;
                    StatusLine = $"Chunk {p.ChunkIndex}/{p.ChunkCount} — {p.TensorsTracked} tensors tracked";
                });

                return await LlamaImatrix.ComputeAsync(
                    model,
                    CorpusText,
                    OutputPath,
                    new LlamaImatrixOptions
                    {
                        ContextSize    = ContextSize,
                        ProcessOutput  = ProcessOutput,
                        ThreadCount    = ThreadCount,
                        DatasetNames   = string.IsNullOrEmpty(CorpusPath)
                            ? null
                            : new[] { Path.GetFileName(CorpusPath) },
                    },
                    progress,
                    _cts.Token);
            }, _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";
            ResultText =
                $"Output:               {OutputPath}\n" +
                $"Tensors tracked:      {result.TensorsTracked:N0}\n" +
                $"Chunks processed:     {result.ChunkCount:N0}\n" +
                $"Tokens processed:     {result.TokensProcessed:N0}\n" +
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
        StatusLine = "Cancellation requested — will take effect at the next chunk boundary.";
    }

    public override void ApplyActiveModel(string? path)
    {
        if (!string.IsNullOrEmpty(path) && string.IsNullOrEmpty(ModelPath)) ModelPath = path;
    }
}
