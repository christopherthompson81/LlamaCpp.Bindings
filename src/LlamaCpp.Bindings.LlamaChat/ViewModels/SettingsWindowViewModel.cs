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
    public McpSettingsViewModel McpSettings { get; } = new();
    public ServerSettingsViewModel ServerSettings { get; } = new();

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
    private async Task DeleteProfileAsync()
    {
        if (SelectedProfile is null) return;
        var target = SelectedProfile;
        var name = string.IsNullOrWhiteSpace(target.Name) ? "Untitled profile" : target.Name;
        var choice = await DialogService.ConfirmAsync(
            "Delete profile",
            $"Delete profile \"{name}\"? This removes it from profiles.json.",
            new[]
            {
                ("cancel", "Cancel", false, false),
                ("delete", "Delete", true, true),
            });
        if (choice != "delete") return;

        var idx = Profiles.IndexOf(target);
        Profiles.Remove(target);
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
    private async Task BrowseMmprojPathAsync()
    {
        if (SelectedProfile is null) return;
        var path = await DialogService.PickGgufFileAsync();
        if (!string.IsNullOrEmpty(path)) SelectedProfile.MmprojPath = path;
    }

    [RelayCommand]
    private void ClearMmprojPath()
    {
        if (SelectedProfile is null) return;
        SelectedProfile.MmprojPath = string.Empty;
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

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private async Task AutoConfigureAsync()
    {
        if (SelectedProfile is null) return;
        var path = SelectedProfile.ModelPath;
        if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
        {
            Status = "Auto-configure: set a valid Model path first.";
            return;
        }

        Status = "Auto-configuring — probing model and hardware…";
        try
        {
            // Off the UI thread: the brief llama_model load blocks for
            // hundreds of ms; the ggml device probe blocks briefly too.
            var result = await Task.Run(() => AutoConfigureService.Configure(path));
            SelectedProfile.ApplyAutoConfigure(result);
            Status = "Auto-configure applied. " + result.Explanation;
        }
        catch (System.Exception ex)
        {
            Status = $"Auto-configure failed: {ex.Message}";
        }
    }

    private bool HasSelection() => SelectedProfile is not null;
}
