using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class AppSettingsViewModel : ObservableObject
{
    [ObservableProperty] private AppThemeMode _themeMode;
    [ObservableProperty] private bool _autoScroll;
    [ObservableProperty] private bool _showMessageStats;
    [ObservableProperty] private bool _showReasoningInProgress;
    [ObservableProperty] private bool _highAccessibilityMode;
    [ObservableProperty] private bool _autoTitleNewConversations;
    [ObservableProperty] private bool _stripThinkingFromHistory;

    // Persisted machinery state — not surfaced in the Settings UI but
    // needs to round-trip through ToModel() so window placement and the
    // last-used profile survive a save/load cycle.
    public double? WindowX { get; set; }
    public double? WindowY { get; set; }
    public double? WindowWidth { get; set; }
    public double? WindowHeight { get; set; }
    public bool WindowMaximized { get; set; }
    public string? WindowScreenName { get; set; }
    public double? WindowScreenRelativeX { get; set; }
    public double? WindowScreenRelativeY { get; set; }
    public string? LastProfileName { get; set; }

    public AppSettingsViewModel() : this(new AppSettings()) { }

    public AppSettingsViewModel(AppSettings model)
    {
        _themeMode = model.ThemeMode;
        _autoScroll = model.AutoScroll;
        _showMessageStats = model.ShowMessageStats;
        _showReasoningInProgress = model.ShowReasoningInProgress;
        _highAccessibilityMode = model.HighAccessibilityMode;
        _autoTitleNewConversations = model.AutoTitleNewConversations;
        _stripThinkingFromHistory = model.StripThinkingFromHistory;
        WindowX = model.WindowX;
        WindowY = model.WindowY;
        WindowWidth = model.WindowWidth;
        WindowHeight = model.WindowHeight;
        WindowMaximized = model.WindowMaximized;
        WindowScreenName = model.WindowScreenName;
        WindowScreenRelativeX = model.WindowScreenRelativeX;
        WindowScreenRelativeY = model.WindowScreenRelativeY;
        LastProfileName = model.LastProfileName;
    }

    public AppSettings ToModel() => new()
    {
        ThemeMode = ThemeMode,
        AutoScroll = AutoScroll,
        ShowMessageStats = ShowMessageStats,
        ShowReasoningInProgress = ShowReasoningInProgress,
        HighAccessibilityMode = HighAccessibilityMode,
        AutoTitleNewConversations = AutoTitleNewConversations,
        StripThinkingFromHistory = StripThinkingFromHistory,
        WindowX = WindowX,
        WindowY = WindowY,
        WindowWidth = WindowWidth,
        WindowHeight = WindowHeight,
        WindowMaximized = WindowMaximized,
        WindowScreenName = WindowScreenName,
        WindowScreenRelativeX = WindowScreenRelativeX,
        WindowScreenRelativeY = WindowScreenRelativeY,
        LastProfileName = LastProfileName,
    };

    // Apply theme live when the user flips the radio in Settings; no need
    // for an explicit Save click to see the change.
    partial void OnThemeModeChanged(AppThemeMode value) => ThemeService.Apply(value);
}
