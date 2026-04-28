using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class ProfileBuilderView : UserControl
{
    public ProfileBuilderView()
    {
        InitializeComponent();
    }

    private void OnModePerCategory(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ProfileBuilderViewModel vm)
            vm.Mode = ProfileBuilderViewModel.CampaignMode.PerCategory;
    }

    private void OnModePerLayer(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ProfileBuilderViewModel vm)
            vm.Mode = ProfileBuilderViewModel.CampaignMode.PerLayer;
    }

    private async void OnBrowseSource(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ProfileBuilderViewModel vm) return;
        var picked = await PickFileAsync(open: true, kind: PickKind.Gguf,
            title: "Select source GGUF (F16 recommended)", suggestedName: null);
        if (picked is not null) vm.SourceModelPath = picked;
    }

    private async void OnBrowseCorpus(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ProfileBuilderViewModel vm) return;
        var picked = await PickFileAsync(open: true, kind: PickKind.Any,
            title: "Select calibration corpus", suggestedName: null);
        if (picked is not null) vm.CorpusPath = picked;
    }

    private async void OnBrowseImatrix(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ProfileBuilderViewModel vm) return;
        var picked = await PickFileAsync(open: true, kind: PickKind.Gguf,
            title: "Select imatrix GGUF", suggestedName: null);
        if (picked is not null) vm.ImatrixPath = picked;
    }

    private async void OnBrowseWorkingDir(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ProfileBuilderViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var res = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select working directory (fast disk)",
            AllowMultiple = false,
        });
        var picked = res.FirstOrDefault()?.TryGetLocalPath();
        if (picked is not null) vm.WorkingDirectory = picked;
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ProfileBuilderViewModel vm) return;
        string? suggested = null;
        if (!string.IsNullOrWhiteSpace(vm.SourceModelPath))
        {
            try
            {
                suggested = $"{Path.GetFileNameWithoutExtension(vm.SourceModelPath)}.profile.json";
            }
            catch { /* leave null */ }
        }
        var picked = await PickFileAsync(open: false, kind: PickKind.Json,
            title: "Save sensitivity profile", suggestedName: suggested);
        if (picked is not null) vm.OutputProfilePath = picked;
    }

    private enum PickKind { Gguf, Json, Any }

    private async Task<string?> PickFileAsync(bool open, PickKind kind, string title, string? suggestedName)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;

        var ggufType = new FilePickerFileType("GGUF model") { Patterns = new[] { "*.gguf" } };
        var jsonType = new FilePickerFileType("JSON")        { Patterns = new[] { "*.json" } };

        var (typeFilter, defaultExt, primaryType) = kind switch
        {
            PickKind.Gguf => (new[] { ggufType, FilePickerFileTypes.All }, "gguf", ggufType),
            PickKind.Json => (new[] { jsonType, FilePickerFileTypes.All }, "json", jsonType),
            _             => (new[] { FilePickerFileTypes.All }, "", FilePickerFileTypes.All),
        };

        if (open)
        {
            var res = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = title,
                AllowMultiple = false,
                FileTypeFilter = typeFilter,
            });
            return res.FirstOrDefault()?.TryGetLocalPath();
        }
        else
        {
            var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
            {
                Title = title,
                SuggestedFileName = suggestedName,
                DefaultExtension = defaultExt,
                FileTypeChoices = new[] { primaryType },
            });
            return res?.TryGetLocalPath();
        }
    }
}
