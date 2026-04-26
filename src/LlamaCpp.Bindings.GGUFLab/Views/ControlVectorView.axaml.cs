using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class ControlVectorView : UserControl
{
    public ControlVectorView()
    {
        InitializeComponent();
    }

    private async void OnBrowseModel(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ControlVectorViewModel vm) return;
        var picked = await OpenFileAsync("Select model GGUF", "GGUF model", "*.gguf");
        if (picked is not null) vm.ModelPath = picked;
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ControlVectorViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var ggufType = new FilePickerFileType("Control vector GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save control vector GGUF",
            DefaultExtension = "gguf",
            FileTypeChoices = new[] { ggufType },
        });
        var path = res?.TryGetLocalPath();
        if (path is not null) vm.OutputPath = path;
    }

    private async void OnLoadPositive(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ControlVectorViewModel vm) return;
        var picked = await OpenFileAsync("Select positive prompts file", "Text", "*.txt", "*.md");
        if (picked is not null) await vm.LoadPositiveFromFileAsync(picked);
    }

    private async void OnLoadNegative(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ControlVectorViewModel vm) return;
        var picked = await OpenFileAsync("Select negative prompts file", "Text", "*.txt", "*.md");
        if (picked is not null) await vm.LoadNegativeFromFileAsync(picked);
    }

    private async Task<string?> OpenFileAsync(string title, string typeName, params string[] patterns)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;
        var fileType = new FilePickerFileType(typeName) { Patterns = patterns };
        var res = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = new[] { fileType, FilePickerFileTypes.All },
        });
        return res.FirstOrDefault()?.TryGetLocalPath();
    }
}
