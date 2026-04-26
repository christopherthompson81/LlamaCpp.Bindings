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

    // Default to physical-core count (approximated as logical/2). Two
    // reasons over either llama.cpp's hard-coded 4 or full ProcessorCount:
    //   - llama.cpp's GGML_DEFAULT_N_THREADS=4 leaves big SMT boxes idle
    //     (e.g. 16 logical threads → 25% util).
    //   - Going to full logical count contends for the per-core SIMD unit
    //     on SMT systems and tends to flatten or regress for AVX-heavy
    //     matmul (verified empirically: 16-logical box plateaus at 8).
    // Users on non-SMT chips can dial up; the spinner accepts any value.
    [ObservableProperty]
    private int _threadCount = Math.Max(1, Environment.ProcessorCount / 2);

    [ObservableProperty]
    private string _statusLine = "Idle.";

    [ObservableProperty]
    private string _resultText = string.Empty;

    [ObservableProperty]
    private double _progressFraction;

    [ObservableProperty]
    private string _elapsedText = string.Empty;

    [ObservableProperty]
    private string _etaText = string.Empty;

    /// <summary>
    /// Snapshot of the form values at the moment Run was clicked.
    /// Shown in a panel that stays visible during and after the run
    /// so the user can answer "what did I configure?" without
    /// scrolling back through the form.
    /// </summary>
    [ObservableProperty]
    private string _runParametersText = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isRunning;

    public bool IsIdle => !IsRunning;

    public string LogText => _log.Text;
    public bool HasFullLog => _log.FullLogPath is not null;

    /// <summary>
    /// Throttled, deduped, bounded log surface. Imatrix can spit out
    /// thousands of lines; the naive StringBuilder + per-line
    /// PropertyChanged path froze the UI. <see cref="ThrottledLogBuffer"/>
    /// caps visible runs, batches re-renders, and mirrors everything
    /// to a tail file for "Save log…".
    /// </summary>
    private readonly ThrottledLogBuffer _log = new();
    private CancellationTokenSource? _cts;

    // Run-timing state. _latestProgress captures the most recent chunk
    // tick from the worker thread; the dispatcher timer reads it once
    // a second and updates the elapsed / ETA strings on the UI thread.
    private DateTime? _runStartedAt;
    private (int Idx, int Count)? _latestProgress;
    private Avalonia.Threading.DispatcherTimer? _tickTimer;

    public ImatrixViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
        // Forward the buffer's throttled Text changes to the view.
        _log.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(ThrottledLogBuffer.Text))
            {
                OnPropertyChanged(nameof(LogText));
                OnPropertyChanged(nameof(HasFullLog));
            }
        };
    }

    /// <summary>Save the full (un-deduped, un-truncated) tail to <paramref name="destPath"/>.</summary>
    public Task SaveFullLogAsync(string destPath) => _log.SaveFullLogAsync(destPath);

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

        _log.Clear();
        ResultText = string.Empty;
        ProgressFraction = 0;
        ElapsedText = string.Empty;
        EtaText = string.Empty;
        _latestProgress = null;
        _runStartedAt = DateTime.UtcNow;
        RunParametersText = BuildRunParametersText();

        _cts = new CancellationTokenSource();
        IsRunning = true;
        StatusLine = "Loading model…";

        StartTickTimer();

        var unsubscribe = _logBus.Subscribe(line => _log.Append(line));

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
                    _latestProgress = (p.ChunkIndex, p.ChunkCount);
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
            _log.Append($"[error] {ex}");
        }
        finally
        {
            unsubscribe();
            IsRunning = false;
            _cts?.Dispose();
            _cts = null;
            StopTickTimer();
            // One last update so the final elapsed time lands without
            // waiting for the next tick, and ETA collapses cleanly.
            UpdateTimingDisplay();
            EtaText = string.Empty;
            // Final flush so the visible log shows the tail of the run
            // without waiting for the next throttle tick.
            _log.Stop();
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
        if (string.IsNullOrEmpty(ModelPath)
            && ResolveGgufFromActive(path) is { } resolved)
            ModelPath = resolved;
    }

    protected override bool HasGgufInputValue => !string.IsNullOrEmpty(ModelPath);

    partial void OnModelPathChanged(string value) => NotifyRemediesChanged();

    /// <summary>
    /// True when neither corpus input is filled — surfaces the
    /// load-wikitext remedy card. Hidden once the user picks a file or
    /// pastes anything in the text box (those routes are always faster
    /// than the canonical fetch when the user has their own data).
    /// </summary>
    public bool ShowLoadWikitextRemedy =>
        string.IsNullOrEmpty(CorpusPath) && string.IsNullOrEmpty(CorpusText);

    partial void OnCorpusPathChanged(string value) =>
        OnPropertyChanged(nameof(ShowLoadWikitextRemedy));
    partial void OnCorpusTextChanged(string value) =>
        OnPropertyChanged(nameof(ShowLoadWikitextRemedy));

    private void StartTickTimer()
    {
        StopTickTimer();
        _tickTimer = new Avalonia.Threading.DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(1),
        };
        _tickTimer.Tick += (_, _) => UpdateTimingDisplay();
        _tickTimer.Start();
        UpdateTimingDisplay();
    }

    private void StopTickTimer()
    {
        if (_tickTimer is not null)
        {
            _tickTimer.Stop();
            _tickTimer = null;
        }
    }

    private void UpdateTimingDisplay()
    {
        if (_runStartedAt is not DateTime started)
        {
            ElapsedText = string.Empty;
            EtaText = string.Empty;
            return;
        }
        var elapsed = DateTime.UtcNow - started;
        ElapsedText = FormatHms(elapsed);

        // ETA is a linear extrapolation: time spent so far × remaining-fraction
        // / done-fraction. Loading the model + computing the first chunk
        // dominate before any progress lands, so we only publish ETA once
        // we have at least one completed chunk to anchor against.
        if (_latestProgress is { Idx: > 0, Count: > 0 } p && elapsed.TotalSeconds > 0)
        {
            var remainingFraction = (double)(p.Count - p.Idx) / p.Idx;
            var eta = TimeSpan.FromSeconds(elapsed.TotalSeconds * remainingFraction);
            EtaText = eta.TotalSeconds < 1 ? "<1s" : FormatHms(eta);
        }
        else
        {
            EtaText = "—";
        }
    }

    private static string FormatHms(TimeSpan span)
    {
        if (span.TotalHours >= 1) return $"{(int)span.TotalHours}h {span.Minutes:D2}m {span.Seconds:D2}s";
        if (span.TotalMinutes >= 1) return $"{span.Minutes}m {span.Seconds:D2}s";
        return $"{span.Seconds}s";
    }

    private string BuildRunParametersText()
    {
        var modelLabel  = string.IsNullOrEmpty(ModelPath)  ? "(none)" : Path.GetFileName(ModelPath);
        var corpusLabel = !string.IsNullOrEmpty(CorpusPath)
            ? Path.GetFileName(CorpusPath)
            : (string.IsNullOrEmpty(CorpusText) ? "(none)" : $"pasted ({CorpusText.Length:N0} chars)");
        var outputLabel = string.IsNullOrEmpty(OutputPath) ? "(auto)" : Path.GetFileName(OutputPath);
        var gpuLabel     = GpuLayerCount == -1 ? "all" : GpuLayerCount.ToString();
        var threadsLabel = ThreadCount    == -1 ? "default" : ThreadCount.ToString();
        var outputTensor = ProcessOutput  ? "yes" : "no";
        return
            $"Model:        {modelLabel}\n" +
            $"Corpus:       {corpusLabel}\n" +
            $"Output:       {outputLabel}\n" +
            $"Context:      {ContextSize}\n" +
            $"GPU layers:   {gpuLabel}\n" +
            $"Threads:      {threadsLabel}\n" +
            $"Process out:  {outputTensor}";
    }
}
