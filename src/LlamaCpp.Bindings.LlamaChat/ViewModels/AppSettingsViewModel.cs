using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class AppSettingsViewModel : ObservableObject
{
    [ObservableProperty] private bool _autoScroll;
    [ObservableProperty] private bool _showMessageStats;
    [ObservableProperty] private bool _showReasoningInProgress;

    public AppSettingsViewModel() : this(new AppSettings()) { }

    public AppSettingsViewModel(AppSettings model)
    {
        _autoScroll = model.AutoScroll;
        _showMessageStats = model.ShowMessageStats;
        _showReasoningInProgress = model.ShowReasoningInProgress;
    }

    public AppSettings ToModel() => new()
    {
        AutoScroll = AutoScroll,
        ShowMessageStats = ShowMessageStats,
        ShowReasoningInProgress = ShowReasoningInProgress,
    };
}
