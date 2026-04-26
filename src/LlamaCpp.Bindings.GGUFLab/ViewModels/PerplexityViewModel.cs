using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the Perplexity page: pick a model + corpus, run a chunked
/// forward-pass loop in <see cref="LlamaPerplexity"/>, stream per-chunk
/// progress, show the final number.
/// </summary>
public sealed partial class PerplexityViewModel : ToolPageViewModel
{
    public override string Title => "Perplexity";
    public override string Description =>
        "Score a corpus against a model — chunked forward passes, log-softmax NLL, exp(mean). Pure C#.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    private string _modelPath = string.Empty;

    [ObservableProperty]
    private string _corpusPath = string.Empty;

    /// <summary>
    /// In-line corpus text. The user can paste directly; loading a file
    /// just populates this field. The Run path always reads from here.
    /// </summary>
    [ObservableProperty]
    private string _corpusText = string.Empty;

    [ObservableProperty]
    private int _contextSize = 512;

    [ObservableProperty]
    private bool _scoreSecondHalfOnly = true;

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

    public PerplexityViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    /// <summary>
    /// Read <paramref name="path"/> into <see cref="CorpusText"/> and remember the path.
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

    /// <summary>
    /// Fetch (or read from cache) the wikitext-2 raw test set — the
    /// reference corpus llama.cpp's published perplexity numbers are
    /// computed against. First call downloads ~700 KB of parquet from
    /// HuggingFace and decodes to ~1.18 MB of plain text.
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
            await SetCorpusFromFileAsync(path);
            StatusLine = $"Loaded wikitext-2 test ({CorpusText.Length:N0} chars). Use n_ctx=512 + second-half scoring to compare against published numbers.";
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
            StatusLine = "Pick or paste a corpus first.";
            return;
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
            // Model load can be slow; do it on the background thread.
            var result = await Task.Run(async () =>
            {
                using var model = new LlamaModel(ModelPath, new LlamaModelParameters
                {
                    GpuLayerCount = GpuLayerCount,
                    UseMmap = true,
                });

                StatusLine = "Scoring…";

                var progress = new Progress<LlamaPerplexityProgress>(p =>
                {
                    ProgressFraction = p.ChunkCount > 0
                        ? (double)p.ChunkIndex / p.ChunkCount
                        : 0;
                    StatusLine = $"Chunk {p.ChunkIndex}/{p.ChunkCount} — running PPL = {p.RunningPerplexity:F3}";
                });

                return await LlamaPerplexity.ComputeAsync(
                    model,
                    CorpusText,
                    new LlamaPerplexityOptions
                    {
                        ContextSize          = ContextSize,
                        ScoreSecondHalfOnly  = ScoreSecondHalfOnly,
                        ThreadCount          = ThreadCount,
                    },
                    progress,
                    _cts.Token);
            }, _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";
            ResultText =
                $"Perplexity:           {result.Perplexity:F4}\n" +
                $"Mean NLL:             {result.NegativeLogLikelihood:F4}\n" +
                $"Tokens scored:        {result.TokensScored:N0}\n" +
                $"Chunks:               {result.ChunkCount}\n" +
                $"Effective context:    {result.ContextSize}\n" +
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
}
