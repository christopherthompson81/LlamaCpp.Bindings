using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LlamaCpp.Bindings.GGUFLab.Services;

/// <summary>
/// Append-only log surface for noisy native pipelines (imatrix collection,
/// perplexity sweeps, anything that emits thousands of lines per run).
/// Solves three problems the naive <c>StringBuilder + OnPropertyChanged</c>
/// pattern hits:
/// <list type="bullet">
///   <item>UI freezes — every line was forcing a full re-render of a
///         growing TextBlock. The buffer batches updates onto a
///         repeating dispatcher tick (default 150 ms) so the visible
///         <see cref="Text"/> changes at a sane cadence regardless of
///         how fast lines arrive.</item>
///   <item>Repeated lines — consecutive identical lines fold into a
///         single entry with a "(×N)" suffix so the visible log keeps
///         signal density.</item>
///   <item>Unbounded growth — the visible buffer caps at
///         <see cref="MaxLines"/> *runs* so memory stays flat. The full,
///         un-deduped, un-truncated tail is mirrored to a temp file
///         that <see cref="SaveFullLogAsync"/> can copy out.</item>
/// </list>
/// </summary>
public sealed partial class ThrottledLogBuffer : ObservableObject, IDisposable
{
    /// <summary>How many consecutive-deduped runs to keep visible.</summary>
    public int MaxLines { get; init; } = 500;

    /// <summary>Cadence of <see cref="Text"/> property notifications.</summary>
    public TimeSpan FlushInterval { get; init; } = TimeSpan.FromMilliseconds(150);

    private readonly LinkedList<Run> _runs = new();
    private string _cachedText = string.Empty;
    private bool _dirty;

    private DispatcherTimer? _timer;
    private string? _fullLogPath;
    private StreamWriter? _fullLogWriter;

    public string Text => _cachedText;

    /// <summary>Path to the temp file with the un-deduped, un-truncated tail. Null if nothing logged yet.</summary>
    public string? FullLogPath => _fullLogPath;

    public void Append(string line)
    {
        // Consecutive-dedup against the tail run.
        if (_runs.Last is { } tail && tail.Value.Line == line)
        {
            tail.Value.Count++;
        }
        else
        {
            _runs.AddLast(new Run { Line = line, Count = 1 });
            while (_runs.Count > MaxLines) _runs.RemoveFirst();
        }
        EnsureLogFile();
        _fullLogWriter?.WriteLine(line);
        _dirty = true;
        EnsureTimer();
    }

    /// <summary>
    /// Stops the throttling timer and flushes the visible buffer + the
    /// tail file once. Safe to call multiple times. Re-arms automatically
    /// on the next <see cref="Append"/>.
    /// </summary>
    public void Stop()
    {
        if (_timer is not null)
        {
            _timer.Stop();
            _timer = null;
        }
        FlushIfDirty();
        _fullLogWriter?.Flush();
    }

    public void Clear()
    {
        Stop();
        _runs.Clear();
        _cachedText = string.Empty;
        OnPropertyChanged(nameof(Text));
        // Reset the tail file too — a fresh run shouldn't carry over the
        // previous run's lines into the saved log.
        try
        {
            _fullLogWriter?.Dispose();
            if (_fullLogPath is not null && File.Exists(_fullLogPath))
                File.Delete(_fullLogPath);
        }
        catch { /* best-effort */ }
        _fullLogWriter = null;
        _fullLogPath = null;
    }

    /// <summary>Copy the un-deduped tail file to <paramref name="destPath"/>.</summary>
    public async Task SaveFullLogAsync(string destPath)
    {
        // Final flush so anything still buffered lands on disk before copy.
        Stop();
        if (_fullLogPath is not null && File.Exists(_fullLogPath))
        {
            File.Copy(_fullLogPath, destPath, overwrite: true);
            return;
        }
        // No tail file yet — best-effort write whatever we have visible.
        await File.WriteAllTextAsync(destPath, _cachedText);
    }

    public void Dispose()
    {
        Stop();
        _fullLogWriter?.Dispose();
        _fullLogWriter = null;
    }

    private void EnsureTimer()
    {
        if (_timer is not null) return;
        _timer = new DispatcherTimer { Interval = FlushInterval };
        _timer.Tick += (_, _) => FlushIfDirty();
        _timer.Start();
    }

    private void FlushIfDirty()
    {
        if (!_dirty) return;
        _dirty = false;
        _cachedText = BuildVisibleText();
        OnPropertyChanged(nameof(Text));
    }

    private void EnsureLogFile()
    {
        if (_fullLogWriter is not null) return;
        try
        {
            var dir = Path.Combine(Path.GetTempPath(), "GGUFLab");
            Directory.CreateDirectory(dir);
            _fullLogPath = Path.Combine(dir,
                $"log-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.log");
            _fullLogWriter = new StreamWriter(_fullLogPath, append: false)
            {
                AutoFlush = false,  // we flush on Stop / Save
            };
        }
        catch
        {
            // If we can't open a tail file (no temp space, etc.) the
            // visible buffer still works — just lose Save fidelity.
            _fullLogPath = null;
            _fullLogWriter = null;
        }
    }

    private string BuildVisibleText()
    {
        if (_runs.Count == 0) return string.Empty;
        var sb = new StringBuilder();
        foreach (var run in _runs)
        {
            sb.Append(run.Line);
            if (run.Count > 1) sb.Append($"  (×{run.Count})");
            sb.Append('\n');
        }
        return sb.ToString();
    }

    private sealed class Run
    {
        public string Line { get; set; } = "";
        public int Count { get; set; }
    }
}
