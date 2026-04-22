namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// Application-wide preferences that aren't tied to a specific model profile.
/// Persisted as a single JSON file by <see cref="Services.AppSettingsStore"/>.
/// </summary>
public sealed record AppSettings
{
    /// <summary>
    /// Keep the message list pinned to the bottom as new content streams in.
    /// Disable this to scroll freely while the assistant is still speaking.
    /// </summary>
    public bool AutoScroll { get; init; } = true;

    /// <summary>Show the tok/s + token-count footer on each assistant bubble.</summary>
    public bool ShowMessageStats { get; init; } = true;

    /// <summary>
    /// Auto-expand the reasoning panel while the assistant is mid-stream, so
    /// you can watch the thinking-model trace unfold in real time.
    /// </summary>
    public bool ShowReasoningInProgress { get; init; } = false;
}
