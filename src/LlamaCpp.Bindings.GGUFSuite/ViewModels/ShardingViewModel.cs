using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.GGUFSuite.Services;

namespace LlamaCpp.Bindings.GGUFSuite.ViewModels;

/// <summary>
/// Operation mode for the Sharding page — selects between split (one file → many)
/// and merge (many files → one).
/// </summary>
public enum ShardingMode
{
    Split,
    Merge,
}

/// <summary>
/// Drives the Sharding page: split a single GGUF into N shards by max
/// tensors-per-shard or max bytes-per-shard, or merge a sharded set
/// back into one file.
/// </summary>
public sealed partial class ShardingViewModel : ToolPageViewModel
{
    public override string Title => "GGUF Sharding";
    public override string Description =>
        "Split a GGUF into shards or merge them back. Pure C# — streams tensor data so multi-GB shards never sit in memory.";

    private readonly NativeLogBus _logBus;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSplitMode))]
    [NotifyPropertyChangedFor(nameof(IsMergeMode))]
    private ShardingMode _mode = ShardingMode.Split;

    public IReadOnlyList<ShardingMode> AvailableModes { get; } =
        new[] { ShardingMode.Split, ShardingMode.Merge };

    public bool IsSplitMode => Mode == ShardingMode.Split;
    public bool IsMergeMode => Mode == ShardingMode.Merge;

    // ----- Split mode fields -----

    [ObservableProperty] private string _splitInputPath = string.Empty;

    /// <summary>Path prefix (no extension); shards expand to <c>&lt;prefix&gt;-NNNNN-of-NNNNN.gguf</c>.</summary>
    [ObservableProperty] private string _splitOutputPrefix = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsTensorsLimitMode))]
    [NotifyPropertyChangedFor(nameof(IsBytesLimitMode))]
    private SplitLimit _splitLimitMode = SplitLimit.Tensors;

    public IReadOnlyList<SplitLimit> AvailableLimitModes { get; } =
        new[] { SplitLimit.Tensors, SplitLimit.Bytes };

    public bool IsTensorsLimitMode => SplitLimitMode == SplitLimit.Tensors;
    public bool IsBytesLimitMode   => SplitLimitMode == SplitLimit.Bytes;

    [ObservableProperty] private int _maxTensorsPerShard = 128;

    /// <summary>Max bytes per shard, in MB (gets multiplied to bytes when running).</summary>
    [ObservableProperty] private int _maxMegabytesPerShard = 4096;

    // ----- Merge mode fields -----

    [ObservableProperty] private string _mergeFirstShardPath = string.Empty;
    [ObservableProperty] private string _mergeOutputPath = string.Empty;

    // ----- Shared run state -----

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

    public ShardingViewModel(NativeLogBus logBus)
    {
        _logBus = logBus;
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        if (IsRunning) return;
        switch (Mode)
        {
            case ShardingMode.Split: await RunSplitAsync(); break;
            case ShardingMode.Merge: await RunMergeAsync(); break;
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        _cts?.Cancel();
        StatusLine = "Cancellation requested.";
    }

    private async Task RunSplitAsync()
    {
        if (string.IsNullOrWhiteSpace(SplitInputPath) || !File.Exists(SplitInputPath))
        {
            StatusLine = "Pick a source GGUF first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(SplitOutputPrefix))
        {
            // Derive: <input-dir>/<input-stem>
            try
            {
                var dir = Path.GetDirectoryName(SplitInputPath) ?? string.Empty;
                var stem = Path.GetFileNameWithoutExtension(SplitInputPath);
                SplitOutputPrefix = Path.Combine(dir, stem);
            }
            catch
            {
                StatusLine = "Pick an output prefix.";
                return;
            }
        }

        var options = new LlamaGgufSplitOptions();
        if (SplitLimitMode == SplitLimit.Tensors)
        {
            options.MaxTensorsPerShard = MaxTensorsPerShard;
        }
        else
        {
            options.MaxBytesPerShard = (long)MaxMegabytesPerShard * 1024 * 1024;
        }

        await RunWithBoilerplate("Splitting", async (progress, ct) =>
        {
            var result = await LlamaGgufSharding.SplitAsync(
                SplitInputPath, SplitOutputPrefix, options, progress, ct);
            ResultText =
                $"Shards:               {result.ShardCount}\n" +
                $"Total bytes:          {result.TotalBytes:N0}\n" +
                $"Elapsed:              {result.Elapsed.TotalSeconds:F2}s\n" +
                $"Output:\n  " + string.Join("\n  ", result.ShardPaths);
            StatusLine = $"Wrote {result.ShardCount} shards in {result.Elapsed.TotalSeconds:F1}s.";
        });
    }

    private async Task RunMergeAsync()
    {
        if (string.IsNullOrWhiteSpace(MergeFirstShardPath) || !File.Exists(MergeFirstShardPath))
        {
            StatusLine = "Pick the first shard (e.g. *-00001-of-NNNNN.gguf).";
            return;
        }
        if (string.IsNullOrWhiteSpace(MergeOutputPath))
        {
            StatusLine = "Pick an output GGUF path.";
            return;
        }

        await RunWithBoilerplate("Merging", async (progress, ct) =>
        {
            var result = await LlamaGgufSharding.MergeAsync(
                MergeFirstShardPath, MergeOutputPath, progress, ct);
            ResultText =
                $"Shards merged:        {result.ShardCount}\n" +
                $"Input bytes:          {result.TotalBytes:N0}\n" +
                $"Output:               {result.OutputPath}\n" +
                $"Output bytes:         {new FileInfo(result.OutputPath).Length:N0}\n" +
                $"Elapsed:              {result.Elapsed.TotalSeconds:F2}s";
            StatusLine = $"Merged {result.ShardCount} shards in {result.Elapsed.TotalSeconds:F1}s.";
        });
    }

    private async Task RunWithBoilerplate(
        string phaseLabel,
        Func<IProgress<LlamaGgufShardingProgress>, CancellationToken, Task> body)
    {
        _logBuilder.Clear();
        OnPropertyChanged(nameof(LogText));
        ResultText = string.Empty;
        ProgressFraction = 0;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        StatusLine = $"{phaseLabel}…";

        var unsubscribe = _logBus.Subscribe(line =>
        {
            _logBuilder.AppendLine(line);
            OnPropertyChanged(nameof(LogText));
        });

        try
        {
            var progress = new Progress<LlamaGgufShardingProgress>(p =>
            {
                if (p.Count > 0)
                {
                    ProgressFraction = (double)p.Index / p.Count;
                }
                StatusLine = $"{phaseLabel}: {p.Phase} — {p.CurrentTensorName}";
            });
            await body(progress, _cts.Token);
            ProgressFraction = 1;
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
}

public enum SplitLimit
{
    Tensors,
    Bytes,
}
