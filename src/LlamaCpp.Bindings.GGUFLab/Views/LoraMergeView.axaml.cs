using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFLab.ViewModels;

namespace LlamaCpp.Bindings.GGUFLab.Views;

public partial class LoraMergeView : UserControl
{
    public LoraMergeView()
    {
        InitializeComponent();
    }

    private async void OnBrowseBase(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not LoraMergeViewModel vm) return;
        var picked = await OpenGgufAsync("Select base model GGUF");
        if (picked is not null) vm.BasePath = picked;
    }

    private async void OnBrowseAdapter(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not LoraMergeViewModel vm) return;
        var picked = await OpenGgufAsync("Select LoRA adapter GGUF");
        if (picked is not null) vm.AdapterPath = picked;
    }

    private async void OnBrowseOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not LoraMergeViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var ggufType = new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save merged GGUF",
            DefaultExtension = "gguf",
            FileTypeChoices = new[] { ggufType },
        });
        var picked = res?.TryGetLocalPath();
        if (picked is not null) vm.OutputPath = picked;
    }

    private async Task<string?> OpenGgufAsync(string title)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return null;
        var ggufType = new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = new[] { ggufType, FilePickerFileTypes.All },
        });
        return res.FirstOrDefault()?.TryGetLocalPath();
    }
}
