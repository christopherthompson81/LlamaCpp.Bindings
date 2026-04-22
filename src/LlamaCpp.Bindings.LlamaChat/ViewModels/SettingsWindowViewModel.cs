using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

public partial class SettingsWindowViewModel : ObservableObject
{
    public ObservableCollection<ProfileEditorViewModel> Profiles { get; }
    public AppSettingsViewModel AppSettings { get; }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DeleteProfileCommand), nameof(DuplicateProfileCommand))]
    private ProfileEditorViewModel? _selectedProfile;

    [ObservableProperty] private string _status = string.Empty;

    public SettingsWindowViewModel(
        ObservableCollection<ProfileEditorViewModel> profiles,
        AppSettingsViewModel appSettings)
    {
        Profiles = profiles;
        AppSettings = appSettings;
        SelectedProfile = profiles.Count > 0 ? profiles[0] : null;
    }

    [RelayCommand]
    private void AddProfile()
    {
        var created = new ProfileEditorViewModel { Name = $"Profile {Profiles.Count + 1}" };
        Profiles.Add(created);
        SelectedProfile = created;
    }

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private void DuplicateProfile()
    {
        if (SelectedProfile is null) return;
        var clone = new ProfileEditorViewModel(SelectedProfile.ToProfile())
        {
            Name = SelectedProfile.Name + " (copy)",
        };
        Profiles.Add(clone);
        SelectedProfile = clone;
    }

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private void DeleteProfile()
    {
        if (SelectedProfile is null) return;
        var idx = Profiles.IndexOf(SelectedProfile);
        Profiles.Remove(SelectedProfile);
        SelectedProfile = Profiles.Count == 0
            ? null
            : Profiles[System.Math.Min(idx, Profiles.Count - 1)];
    }

    [RelayCommand]
    private async Task BrowseModelPathAsync()
    {
        if (SelectedProfile is null) return;
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) SelectedProfile.ModelPath = path;
    }

    [RelayCommand]
    private void Save()
    {
        try
        {
            ProfileStore.Save(Profiles.Select(p => p.ToProfile()));
            AppSettingsStore.Save(AppSettings.ToModel());
            Status = $"Saved {Profiles.Count} profile(s) + app settings.";
        }
        catch (System.Exception ex)
        {
            Status = $"Save failed: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ResetSamplerDefaults()
    {
        SelectedProfile?.ResetSamplerDefaults();
        Status = "Reset sampling to defaults.";
    }

    private bool HasSelection() => SelectedProfile is not null;
}
