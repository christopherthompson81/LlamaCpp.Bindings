using System.Linq;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFSuite.ViewModels;

namespace LlamaCpp.Bindings.GGUFSuite.Views;

public partial class HfConvertView : UserControl
{
    public HfConvertView()
    {
        InitializeComponent();
    }

    private async void OnBrowseHfDirectory(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not HfConvertViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var res = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select HuggingFace model directory",
            AllowMultiple = false,
        });
        var path = res.FirstOrDefault()?.TryGetLocalPath();
        if (path is not null) vm.HfDirectory = path;
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not HfConvertViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var ggufType = new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save GGUF",
            DefaultExtension = "gguf",
            FileTypeChoices = new[] { ggufType },
        });
        var path = res?.TryGetLocalPath();
        if (path is not null) vm.OutputPath = path;
    }
}
