using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFSuite.ViewModels;

namespace LlamaCpp.Bindings.GGUFSuite.Views;

public partial class AdaptiveQuantizeView : UserControl
{
    public AdaptiveQuantizeView()
    {
        InitializeComponent();
    }

    private async void OnBrowseInput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: true, ggufOnly: true,
            title: "Select source GGUF", suggestedName: null);
        if (picked is not null) vm.InputPath = picked;
    }

    private async void OnBrowseImatrix(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: true, ggufOnly: true,
            title: "Select imatrix GGUF", suggestedName: null);
        if (picked is not null) vm.ImatrixPath = picked;
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
                suggested = $"{stem}.adaptive-{vm.Threshold:F3}.gguf";
            }
            catch { /* leave null */ }
        }
        var picked = await PickFileAsync(open: false, ggufOnly: true,
            title: "Save quantized GGUF", suggestedName: suggested);
        if (picked is not null) vm.OutputPath = picked;
    }

    private async void OnSaveScores(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: false, ggufOnly: false,
            title: "Save sensitivity score table",
            suggestedName: SuggestedJsonName(vm.InputPath, "scores"));
        if (picked is not null) vm.SaveScoresJson(picked);
    }

    private async void OnLoadScores(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: true, ggufOnly: false,
            title: "Load sensitivity score table", suggestedName: null);
        if (picked is not null) vm.LoadScoresJson(picked);
    }

    private async void OnSaveRecipe(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not AdaptiveQuantizeViewModel vm) return;
        var picked = await PickFileAsync(open: false, ggufOnly: false,
            title: "Save quantization recipe",
            suggestedName: SuggestedJsonName(vm.InputPath, $"recipe-{vm.Threshold:F3}"));
        if (picked is not null) vm.SaveRecipeJson(picked);
    }

    private static string? SuggestedJsonName(string inputPath, string suffix)
    {
        if (string.IsNullOrWhiteSpace(inputPath)) return null;
        try
        {
            return $"{Path.GetFileNameWithoutExtension(inputPath)}.{suffix}.json";
        }
        catch
        {
            return null;
        }
    }

    private async Task<string?> PickFileAsync(bool open, bool ggufOnly, string title, string? suggestedName)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;

        var ggufType = new FilePickerFileType("GGUF model") { Patterns = new[] { "*.gguf" } };
        var jsonType = new FilePickerFileType("JSON")        { Patterns = new[] { "*.json" } };
        var typeFilter = ggufOnly
            ? new[] { ggufType, FilePickerFileTypes.All }
            : new[] { jsonType, FilePickerFileTypes.All };

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
                DefaultExtension = ggufOnly ? "gguf" : "json",
                FileTypeChoices = ggufOnly ? new[] { ggufType } : new[] { jsonType },
            });
            return res?.TryGetLocalPath();
        }
    }
}
