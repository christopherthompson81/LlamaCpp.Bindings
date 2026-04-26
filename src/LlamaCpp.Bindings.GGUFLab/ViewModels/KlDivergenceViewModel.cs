using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFLab.Services;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the KL Divergence page: pick a reference and a test model
/// (typically baseline F16 vs quantized variant), pick a corpus, run
/// <see cref="LlamaKlDivergence"/>, report the canonical "did the
/// quant preserve quality?" metrics.
/// </summary>
public sealed partial class KlDivergenceViewModel : ToolPageViewModel
{
    public override string Title => "KL Divergence";
    public override string Description =>
        "Compare a quantized variant against its baseline (e.g. Q4_K_M vs F16) on a shared corpus. Reports KL, top-K agreement, and ΔPPL.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty] private string _referenceModelPath = string.Empty;
    [ObservableProperty] private string _testModelPath = string.Empty;

    [ObservableProperty] private string _corpusPath = string.Empty;
    [ObservableProperty] private string _corpusText = string.Empty;

    [ObservableProperty] private int _contextSize = 512;
    [ObservableProperty] private bool _scoreSecondHalfOnly = true;
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

    public KlDivergenceViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

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
                    StatusLine = $"Downloading wikitext-2… {p.downloaded / 1024:N0} / {t / 1024:N0} KB ({pct}%)";
                }
            });
            var path = await WikitextCorpus.EnsureTestRawAsync(progress);
            await SetCorpusFromFileAsync(path);
            StatusLine = $"Loaded wikitext-2 test ({CorpusText.Length:N0} chars).";
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
        if (string.IsNullOrWhiteSpace(ReferenceModelPath) || !File.Exists(ReferenceModelPath))
        {
            StatusLine = "Pick a reference model GGUF (typically F16/F32 baseline).";
            return;
        }
        if (string.IsNullOrWhiteSpace(TestModelPath) || !File.Exists(TestModelPath))
        {
            StatusLine = "Pick a test model GGUF (typically the quantized variant).";
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
        StatusLine = "Loading models…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var result = await Task.Run(async () =>
            {
                using var refModel = new LlamaModel(ReferenceModelPath, new LlamaModelParameters
                {
                    GpuLayerCount = GpuLayerCount,
                    UseMmap = true,
                });
                using var testModel = new LlamaModel(TestModelPath, new LlamaModelParameters
                {
                    GpuLayerCount = GpuLayerCount,
                    UseMmap = true,
                });

                StatusLine = "Computing KL…";
                var progress = new Progress<LlamaKlDivergenceProgress>(p =>
                {
                    ProgressFraction = p.ChunkCount > 0
                        ? (double)p.ChunkIndex / p.ChunkCount
                        : 0;
                    StatusLine = $"Chunk {p.ChunkIndex}/{p.ChunkCount} — running mean KL = {p.RunningMeanKl:F6}";
                });

                return await LlamaKlDivergence.ComputeAsync(
                    refModel, testModel, CorpusText,
                    new LlamaKlDivergenceOptions
                    {
                        ContextSize         = ContextSize,
                        ScoreSecondHalfOnly = ScoreSecondHalfOnly,
                        ThreadCount         = ThreadCount,
                    },
                    progress,
                    _cts.Token);
            }, _cts.Token);

            ProgressFraction = 1;
            StatusLine = $"Done in {result.Elapsed.TotalSeconds:F1}s.";

            // Convert nats → bits for the human-friendly summary line;
            // researchers and llama.cpp's tool both quote KL in bits.
            const double Ln2 = 0.69314718055994530941723212145818;
            ResultText =
                $"Mean KL:           {result.MeanKl:F6} nats  ({result.MeanKl / Ln2:F6} bits)\n" +
                $"Median KL:         {result.MedianKl:F6} nats\n" +
                $"P90 KL:            {result.P90Kl:F6} nats\n" +
                $"P99 KL:            {result.P99Kl:F6} nats\n" +
                $"Max KL:            {result.MaxKl:F6} nats\n" +
                $"\n" +
                $"Top-1 agreement:   {result.Top1AgreementRate * 100:F2}%\n" +
                $"Top-5 agreement:   {result.Top5AgreementRate * 100:F2}%\n" +
                $"\n" +
                $"Reference PPL:     {result.ReferencePerplexity:F4}\n" +
                $"Test PPL:          {result.TestPerplexity:F4}\n" +
                $"ΔPPL:              {result.TestPerplexity - result.ReferencePerplexity:+0.0000;-0.0000;0.0000}  " +
                $"({(result.ReferencePerplexity > 0 ? (result.TestPerplexity / result.ReferencePerplexity - 1) * 100 : 0):+0.00;-0.00;0.00}%)\n" +
                $"\n" +
                $"Tokens scored:     {result.TokensScored:N0}\n" +
                $"Chunks:            {result.ChunkCount:N0}\n" +
                $"Effective context: {result.ContextSize}\n" +
                $"Elapsed:           {result.Elapsed.TotalSeconds:F2}s";
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
        // KL has two slots; the active model is the candidate being
        // judged against the reference, so it lands in TestModelPath.
        if (string.IsNullOrEmpty(TestModelPath)
            && ResolveGgufFromActive(path) is { } resolved)
            TestModelPath = resolved;
    }

    // The safetensors-conversion remedy is offered against the test
    // model slot since that's where ApplyActiveModel routes the active
    // path. Reference is typically a fixed baseline GGUF the user owns.
    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(TestModelPath);

    partial void OnTestModelPathChanged(string value) => NotifyRemediesChanged();
}
