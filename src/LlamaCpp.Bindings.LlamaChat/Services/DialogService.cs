using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;
using LlamaCpp.Bindings.LlamaChat.ViewModels;
using LlamaCpp.Bindings.LlamaChat.Views;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Narrow host for things that need the main <see cref="Window"/> as owner —
/// dialog windows and file-picker invocations. Kept as static helpers so the
/// view models don't have to take a UI service dependency.
/// </summary>
internal static class DialogService
{
    private static Window? Owner =>
        (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow;

    public static async Task ShowSettingsAsync(SettingsWindowViewModel vm)
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new SettingsWindow { DataContext = vm };
        await win.ShowDialog(owner);
    }

    public static async Task<string?> PickGgufFileAsync()
    {
        var owner = Owner;
        if (owner is null) return null;
        var result = await owner.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select model (.gguf)",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("GGUF models") { Patterns = new[] { "*.gguf" } },
                FilePickerFileTypes.All,
            },
        });
        return result.Count > 0 ? result[0].TryGetLocalPath() : null;
    }
}
