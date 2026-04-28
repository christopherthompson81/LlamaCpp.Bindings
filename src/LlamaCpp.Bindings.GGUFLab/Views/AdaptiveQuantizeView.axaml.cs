using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class AdaptiveQuantizeView : UserControl
{
    public AdaptiveQuantizeView()
    {
        InitializeComponent();
    }

    private async void OnBrowseInput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: true, filter: FileFilter.Gguf,
            title: "Select source GGUF", suggestedName: null);
        if (picked is not null) vm.InputPath = picked;
    }

    private async void OnBrowseImatrix(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: true, filter: FileFilter.Gguf,
            title: "Select imatrix GGUF", suggestedName: null);
        if (picked is not null) vm.ImatrixPath = picked;
    }

    private async void OnBrowseProfile(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: true, filter: FileFilter.Json,
            title: "Select sensitivity profile (.profile.json)", suggestedName: null);
        if (picked is not null) vm.ProfilePath = picked;
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        string? suggested = null;
        if (!string.IsNullOrWhiteSpace(vm.InputPath))
        {
            try
            {
                var stem = Path.GetFileNameWithoutExtension(vm.InputPath);
                suggested = $"{stem}.profile-{vm.TargetBitsPerElement:F2}.gguf";
            }
            catch { /* leave null */ }
        }
        var picked = await PickFileAsync(open: false, filter: FileFilter.Gguf,
            title: "Save quantized GGUF", suggestedName: suggested);
        if (picked is not null) vm.OutputPath = picked;
    }

    private async void OnSaveRecipe(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        string? suggested = null;
        if (!string.IsNullOrWhiteSpace(vm.InputPath))
        {
            try
            {
                suggested = $"{Path.GetFileNameWithoutExtension(vm.InputPath)}.recipe-{vm.TargetBitsPerElement:F2}.json";
            }
            catch { /* leave null */ }
        }
        var picked = await PickFileAsync(open: false, filter: FileFilter.Json,
            title: "Save quantization recipe", suggestedName: suggested);
        if (picked is not null) vm.SaveRecipeJson(picked);
    }

    private void OnPresetQ4KM(object? sender, RoutedEventArgs e)
    {
        if (DataContext is AdaptiveQuantizeViewModel vm) vm.TargetBitsPerElement = 4.95;
    }

    private void OnPresetQ5KM(object? sender, RoutedEventArgs e)
    {
        if (DataContext is AdaptiveQuantizeViewModel vm) vm.TargetBitsPerElement = 5.50;
    }

    private void OnPresetQ6K(object? sender, RoutedEventArgs e)
    {
        if (DataContext is AdaptiveQuantizeViewModel vm) vm.TargetBitsPerElement = 6.5625;
    }

    private void OnPresetQ80(object? sender, RoutedEventArgs e)
    {
        if (DataContext is AdaptiveQuantizeViewModel vm) vm.TargetBitsPerElement = 8.50;
    }

    private enum FileFilter { Gguf, Json }

    private async Task<string?> PickFileAsync(bool open, FileFilter filter, string title, string? suggestedName)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;

        var ggufType = new FilePickerFileType("GGUF model") { Patterns = new[] { "*.gguf" } };
        var jsonType = new FilePickerFileType("JSON")        { Patterns = new[] { "*.json" } };
        var (typeFilter, defaultExt, primaryType) = filter switch
        {
            FileFilter.Gguf => (new[] { ggufType, FilePickerFileTypes.All }, "gguf", ggufType),
            FileFilter.Json => (new[] { jsonType, FilePickerFileTypes.All }, "json", jsonType),
            _               => (new[] { FilePickerFileTypes.All }, "", FilePickerFileTypes.All),
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
