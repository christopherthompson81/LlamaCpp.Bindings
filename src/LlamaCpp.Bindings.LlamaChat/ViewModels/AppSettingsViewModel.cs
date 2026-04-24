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
    };

    // Apply theme live when the user flips the radio in Settings; no need
    // for an explicit Save click to see the change.
    partial void OnThemeModeChanged(AppThemeMode value) => ThemeService.Apply(value);
}
