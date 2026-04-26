using System;
using System.Collections.Generic;
using Avalonia.Threading;

namespace LlamaCpp.Bindings.GGUFSuite.Services;

/// <summary>
/// Tiny pub-sub for native llama.cpp log lines. Installed once at app
/// startup as the <see cref="LlamaBackend"/> log sink; pages subscribe for
/// the duration of an operation and unsubscribe when done.
/// </summary>
/// <remarks>
/// llama.cpp invokes its log callback from native threads, so we marshal
/// every line onto the Avalonia UI thread before fanning out to subscribers.
/// That way subscribers can mutate <c>ObservableObject</c> properties without
/// needing their own <c>Dispatcher</c> hop.
/// </remarks>
public sealed class NativeLogBus
{
    private readonly object _gate = new();
    private readonly List<Action<string>> _subscribers = new();

    /// <summary>
    /// The callback to register with <see cref="LlamaBackend.Initialize"/>.
    /// Captures the level and message into a single line for the log pane.
    /// </summary>
    public void Publish(LlamaLogLevel level, string message)
    {
        var line = level switch
        {
            LlamaLogLevel.Continuation => message,
            _ => $"[{level.ToString().ToLowerInvariant()}] {message}",
        };

        // Snapshot so subscribers added/removed during dispatch don't
        // race the iteration.
        Action<string>[] snapshot;
        lock (_gate)
        {
            if (_subscribers.Count == 0) return;
            snapshot = _subscribers.ToArray();
        }

        // Marshal to UI thread; logs cross back from native worker threads.
        Dispatcher.UIThread.Post(() =>
        {
            foreach (var s in snapshot) s(line);
        });
    }

    /// <summary>
    /// Adds a subscriber. Returns an unsubscribe action — call it when the
    /// subscriber's lifetime ends (typically <c>finally</c> after a run).
    /// </summary>
    public Action Subscribe(Action<string> sink)
    {
        lock (_gate) _subscribers.Add(sink);
        return () =>
        {
            lock (_gate) _subscribers.Remove(sink);
        };
    }
}
