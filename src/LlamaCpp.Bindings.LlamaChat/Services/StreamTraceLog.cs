using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Diagnostic trace for the streaming pipeline (token producer →
/// dispatcher posts → UI handler). Off by default; enable by setting
/// the <c>LLAMACHAT_STREAM_TRACE</c> environment variable to <c>1</c>
/// (or anything truthy: <c>true</c>, <c>yes</c>, <c>on</c>) before
/// starting the app.
/// </summary>
/// <remarks>
/// <para>When disabled, <see cref="Log"/> is a no-op early-return that
/// barely costs a static field read; safe to leave at every interesting
/// trace point.</para>
///
/// <para>When enabled, events go to <c>%AppData%/LlamaChat/stream-trace.log</c>
/// (Linux: <c>~/.config/LlamaChat/stream-trace.log</c>) — truncated on
/// app start. Each line has a microsecond timestamp + managed thread id
/// + label, which is enough to diagnose UI-thread vs pool-thread
/// scheduling problems and Dispatcher.Post flow-through.</para>
///
/// <para>Originally added to chase down a freeze where every per-token
/// decode + sample was running on the UI thread (uncontested
/// SemaphoreSlim awaits don't yield), causing dispatcher posts to queue
/// up and never fire until end-of-generation. Keeping the plumbing for
/// the next regression of this shape.</para>
/// </remarks>
public static class StreamTraceLog
{
    public static string LogPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "LlamaChat",
        "stream-trace.log");

    /// <summary>Whether the trace is active for this process. Read once at startup.</summary>
    public static bool IsEnabled { get; }

    private static readonly Stopwatch _sw = new();
    private static readonly object _lock = new();
    private static readonly List<string> _buffer = new(capacity: 4096);

    static StreamTraceLog()
    {
        var raw = Environment.GetEnvironmentVariable("LLAMACHAT_STREAM_TRACE");
        IsEnabled = !string.IsNullOrEmpty(raw)
            && (raw.Equals("1", StringComparison.Ordinal)
                || raw.Equals("true", StringComparison.OrdinalIgnoreCase)
                || raw.Equals("yes", StringComparison.OrdinalIgnoreCase)
                || raw.Equals("on", StringComparison.OrdinalIgnoreCase));
        if (!IsEnabled) return;

        _sw.Start();
        try
        {
            var dir = Path.GetDirectoryName(LogPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(LogPath,
                $"# stream-trace.log — started {DateTime.Now:o}\n" +
                $"# t_us = microseconds since trace start; tid = managed thread id\n" +
                $"# columns: t_us\ttid\tlabel\n");
        }
        catch
        {
            // Best-effort — never let trace setup crash the app.
        }
    }

    /// <summary>
    /// Append a single event. No-op early-return when the env var was
    /// not set, so call sites can stay in production without measurable
    /// cost.
    /// </summary>
    public static void Log(string label)
    {
        if (!IsEnabled) return;
        var us = _sw.Elapsed.TotalMilliseconds * 1000.0;
        var tid = Thread.CurrentThread.ManagedThreadId;
        var line = $"{us:F1}\t{tid}\t{label}\n";
        lock (_lock) _buffer.Add(line);
        MaybeFlush();
    }

    private static long _flushPendingTicks;
    private const long FlushDebounceMs = 250;

    private static void MaybeFlush()
    {
        // Coalesce flushes — batch up writes for ~250 ms so a hot loop
        // doesn't slam the file.
        var now = Environment.TickCount64;
        var prev = Interlocked.Read(ref _flushPendingTicks);
        if (now - prev < FlushDebounceMs) return;
        if (Interlocked.CompareExchange(ref _flushPendingTicks, now, prev) != prev) return;

        // Hand off to a thread-pool task so the producer never blocks
        // on disk I/O.
        _ = System.Threading.Tasks.Task.Run(static () =>
        {
            string toWrite;
            lock (_lock)
            {
                if (_buffer.Count == 0) return;
                var sb = new StringBuilder(_buffer.Count * 64);
                foreach (var l in _buffer) sb.Append(l);
                _buffer.Clear();
                toWrite = sb.ToString();
            }
            try { File.AppendAllText(LogPath, toWrite); }
            catch { /* swallow */ }
        });
    }

    /// <summary>
    /// Force a synchronous flush of all pending entries. Call at the end
    /// of a generation cycle so the trace file is up-to-date when the
    /// user inspects it. No-op when the trace is disabled.
    /// </summary>
    public static void Flush()
    {
        if (!IsEnabled) return;
        string toWrite;
        lock (_lock)
        {
            if (_buffer.Count == 0) return;
            var sb = new StringBuilder(_buffer.Count * 64);
            foreach (var l in _buffer) sb.Append(l);
            _buffer.Clear();
            toWrite = sb.ToString();
        }
        try { File.AppendAllText(LogPath, toWrite); }
        catch { /* swallow */ }
    }
}
