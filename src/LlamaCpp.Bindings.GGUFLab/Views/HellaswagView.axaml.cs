using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class HellaswagView : UserControl
{
    public HellaswagView()
    {
        InitializeComponent();
    }

    private async void OnBrowseModel(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not HellaswagViewModel vm) return;
        var picked = await OpenFileAsync("Select model GGUF", "GGUF model", "*.gguf");
        if (picked is not null) vm.ModelPath = picked;
    }

    private async void OnBrowseDataset(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not HellaswagViewModel vm) return;
        var picked = await OpenFileAsync("Select HellaSwag dataset file", "Text", "*.txt", "*.tsv");
        if (picked is not null) vm.DatasetPath = picked;
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
