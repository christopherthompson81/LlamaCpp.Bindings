using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFSuite.ViewModels;

namespace LlamaCpp.Bindings.GGUFSuite.Views;

public partial class GgufEditorView : UserControl
{
    public GgufEditorView()
    {
        InitializeComponent();
    }

    private async void OnOpen(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not GgufEditorViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var ggufType = new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open GGUF",
            AllowMultiple = false,
            FileTypeFilter = new[] { ggufType, FilePickerFileTypes.All },
        });
        var path = res.FirstOrDefault()?.TryGetLocalPath();
        if (path is not null) await vm.OpenAsync(path);
    }

    private async void OnSaveAs(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not GgufEditorViewModel vm) return;
        if (!vm.IsLoaded) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var ggufType = new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save edited GGUF",
            DefaultExtension = "gguf",
            FileTypeChoices = new[] { ggufType },
        });
        var path = res?.TryGetLocalPath();
        if (path is not null) await vm.SaveAsync(path);
    }

    /// <summary>
    /// Row-button click handler. Each metadata row is rendered as a
    /// transparent <see cref="Button"/> whose <see cref="Button.Tag"/>
    /// holds the row's <see cref="MetadataRow"/> view-model — selecting
    /// it just sets the VM property.
    /// </summary>
    private void OnMetadataRowClick(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not GgufEditorViewModel vm) return;
        if (sender is Button b && b.Tag is MetadataRow row)
        {
            vm.SelectedMetadataRow = row;
        }
    }
}
