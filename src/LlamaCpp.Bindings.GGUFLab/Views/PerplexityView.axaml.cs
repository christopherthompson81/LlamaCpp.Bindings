using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class PerplexityView : UserControl
{
    public PerplexityView()
    {
        InitializeComponent();
    }

    private async void OnBrowseModel(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not PerplexityViewModel vm) return;
        var picked = await PickFileAsync("Select model GGUF", "GGUF model", "*.gguf");
        if (picked is not null) vm.ModelPath = picked;
    }

    private async void OnBrowseCorpus(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not PerplexityViewModel vm) return;
        var picked = await PickFileAsync("Select corpus", "Text", "*.txt", "*.raw", "*.md");
        if (picked is not null)
        {
            await vm.SetCorpusFromFileAsync(picked);
        }
    }

    private async Task<string?> PickFileAsync(string title, string typeName, params string[] patterns)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;

        var fileType = new FilePickerFileType(typeName)
        {
            Patterns = patterns,
        };
        var res = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = new[] { fileType, FilePickerFileTypes.All },
        });
        return res.FirstOrDefault()?.TryGetLocalPath();
    }
}
