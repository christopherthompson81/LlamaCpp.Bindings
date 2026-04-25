namespace LlamaCpp.Bindings.LlamaChat.Models;

public enum AppThemeMode
{
    /// <summary>Follow the system light/dark preference.</summary>
    Auto,
    Light,
    Dark,
}

/// <summary>
/// Application-wide preferences that aren't tied to a specific model profile.
/// Persisted as a single JSON file by <see cref="Services.AppSettingsStore"/>.
/// </summary>
public sealed record AppSettings
{
    /// <summary>Light / Dark / Auto (follow system).</summary>
    public AppThemeMode ThemeMode { get; init; } = AppThemeMode.Auto;

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

    /// <summary>
    /// Heightened accessibility: forces the message action bar visible
    /// (bypasses hover-to-reveal) so every action is reachable without a
    /// pointer, for screen-reader and keyboard-only users.
    /// </summary>
    public bool HighAccessibilityMode { get; init; } = false;

    /// <summary>
    /// When the user sends the first message in a new conversation whose
    /// title is still the default placeholder, set the title from the first
    /// non-empty line of that message (truncated). Matches webui's
    /// "Use first non-empty line for conversation title".
    /// </summary>
    public bool AutoTitleNewConversations { get; init; } = true;

    /// <summary>
    /// When true, assistant messages sent back to the model as prior context
    /// omit their <c>&lt;think&gt;...&lt;/think&gt;</c> blocks. Off preserves
    /// the chain-of-thought across turns — useful for workflows that want
    /// the model to see its prior reasoning, at the cost of context budget.
    /// Matches webui's "Strip thinking from message history".
    /// </summary>
    public bool StripThinkingFromHistory { get; init; } = true;

    /// <summary>
    /// Restored on next launch. Null on first launch (OS chooses default
    /// placement / size). <see cref="WindowX"/>/<see cref="WindowY"/> are
    /// absolute virtual-desktop coordinates and serve as the fallback when
    /// the previous monitor can't be matched by name.
    /// </summary>
    public double? WindowX { get; init; }
    public double? WindowY { get; init; }
    public double? WindowWidth { get; init; }
    public double? WindowHeight { get; init; }
    public bool WindowMaximized { get; init; }

    /// <summary>
    /// <see cref="Avalonia.Platform.Screen.DisplayName"/> of the monitor the
    /// window was on, plus the position relative to that screen's top-left.
    /// On next launch, if a screen with the same display name is still
    /// connected the window is placed on it at the same relative offset —
    /// this survives multi-monitor rearrangement or a single-monitor disconnect
    /// that would otherwise leave the window stranded off-screen.
    /// </summary>
    public string? WindowScreenName { get; init; }
    public double? WindowScreenRelativeX { get; init; }
    public double? WindowScreenRelativeY { get; init; }

    /// <summary>
    /// Name of the profile that was selected the last time the app exited.
    /// Reselected on next launch when present in the profile list; falls
    /// back to the first profile if the named one is gone.
    /// </summary>
    public string? LastProfileName { get; init; }
}
