using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class QuantizeView : UserControl
{
    public QuantizeView()
    {
        InitializeComponent();
    }

    private async void OnBrowseInput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not QuantizeViewModel vm) return;
        var picked = await PickGgufAsync(open: true, suggestedName: null);
        if (picked is not null)
        {
            vm.InputPath = picked;
        }
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not QuantizeViewModel vm) return;
        // Suggest a sensible default name based on the input + ftype so the
        // save dialog opens already populated.
        string? suggested = null;
        if (!string.IsNullOrWhiteSpace(vm.InputPath))
        {
            try
            {
                var stem = Path.GetFileNameWithoutExtension(vm.InputPath);
                suggested = $"{stem}.{vm.SelectedFileType}.gguf";
            }
            catch { /* leave null */ }
        }
        var picked = await PickGgufAsync(open: false, suggestedName: suggested);
        if (picked is not null)
        {
            vm.OutputPath = picked;
        }
    }

    private async Task<string?> PickGgufAsync(bool open, string? suggestedName)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;

        var ggufType = new FilePickerFileType("GGUF model")
        {
            Patterns = new[] { "*.gguf" },
        };

        if (open)
        {
            var res = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Select source GGUF",
                AllowMultiple = false,
                FileTypeFilter = new[] { ggufType, FilePickerFileTypes.All },
            });
            return res.FirstOrDefault()?.TryGetLocalPath();
        }
        else
        {
            var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
            {
                Title = "Save quantized GGUF",
                SuggestedFileName = suggestedName,
                DefaultExtension = "gguf",
                FileTypeChoices = new[] { ggufType },
            });
            return res?.TryGetLocalPath();
        }
    }
}
