using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.GGUFSuite.ViewModels;

namespace LlamaCpp.Bindings.GGUFSuite.Views;

public partial class ShardingView : UserControl
{
    public ShardingView()
    {
        InitializeComponent();
    }

    private async void OnBrowseSplitInput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ShardingViewModel vm) return;
        var picked = await OpenGgufAsync("Select GGUF to split");
        if (picked is not null) vm.SplitInputPath = picked;
    }

    private async void OnBrowseSplitOutputPrefix(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ShardingViewModel vm) return;
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        // We pick a "save" target as if it were a single GGUF, then strip
        // the .gguf extension so the user-visible value is the prefix.
        // Avalonia has no "pick prefix" dialog; this is the cleanest UX.
        var ggufType = new FilePickerFileType("GGUF") { Patterns = new[] { "*.gguf" } };
        var res = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Pick output prefix (the .gguf extension will be stripped)",
            DefaultExtension = "gguf",
            FileTypeChoices = new[] { ggufType },
        });
        var picked = res?.TryGetLocalPath();
        if (picked is null) return;
        // Strip a trailing .gguf if present so the user gets a true prefix.
        if (picked.EndsWith(".gguf", System.StringComparison.OrdinalIgnoreCase))
        {
            picked = picked[..^".gguf".Length];
        }
        vm.SplitOutputPrefix = picked;
    }

    private async void OnBrowseMergeFirst(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ShardingViewModel vm) return;
        var picked = await OpenGgufAsync("Select first shard (e.g. *-00001-of-NNNNN.gguf)");
        if (picked is not null) vm.MergeFirstShardPath = picked;
    }

    private async void OnBrowseMergeOutput(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ShardingViewModel vm) return;
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
        if (picked is not null) vm.MergeOutputPath = picked;
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
