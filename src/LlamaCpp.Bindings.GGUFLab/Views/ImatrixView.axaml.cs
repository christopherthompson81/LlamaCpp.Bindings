using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class ImatrixView : UserControl
{
    public ImatrixView()
    {
        InitializeComponent();
    }

    private async void OnBrowseModel(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ImatrixViewModel vm) return;
        var picked = await OpenFileAsync("Select model GGUF", "GGUF model", "*.gguf");
        if (picked is not null) vm.ModelPath = picked;
    }

    private async void OnBrowseCorpus(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ImatrixViewModel vm) return;
        var picked = await OpenFileAsync("Select calibration corpus", "Text", "*.txt", "*.raw", "*.md");
        if (picked is not null)
        {
            await vm.SetCorpusFromFileAsync(picked);
        }
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ImatrixViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var ggufType = new FilePickerFileType("GGUF imatrix") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save imatrix GGUF",
            DefaultExtension = "gguf",
            FileTypeChoices = new[] { ggufType },
        });
        var path = res?.TryGetLocalPath();
        if (path is not null) vm.OutputPath = path;
    }

    private async void OnSaveLog(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ImatrixViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var logType = new FilePickerFileType("Log file") { Patterns = new[] { "*.log", "*.txt" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save imatrix log",
            DefaultExtension = "log",
            SuggestedFileName = $"imatrix-{System.DateTime.Now:yyyyMMdd-HHmmss}.log",
            FileTypeChoices = new[] { logType },
        });
        var path = res?.TryGetLocalPath();
        if (path is not null) await vm.SaveFullLogAsync(path);
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
