using System.Collections.Generic;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Input.Platform;
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

    public static async Task CopyToClipboardAsync(string? text)
    {
        if (string.IsNullOrEmpty(text)) return;
        var owner = Owner;
        if (owner?.Clipboard is null) return;
        await owner.Clipboard.SetTextAsync(text);
    }

    /// <summary>
    /// Show the given code at full window size in a modal with its own copy
    /// button. The header "Expand" action on each code block in a chat
    /// bubble routes here.
    /// </summary>
    public static async Task ShowShortcutsAsync()
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new ShortcutsDialog();
        await win.ShowDialog(owner);
    }

    public static async Task<string?> ConfirmAsync(
        string title, string message,
        IReadOnlyList<(string Key, string Label, bool Destructive, bool Primary)> options)
    {
        var owner = Owner;
        if (owner is null) return null;
        var win = new ConfirmDialog(title, message, options);
        await win.ShowDialog(owner);
        return win.SelectedKey;
    }

    public static async Task ShowModelInfoAsync(LlamaModel model, string? profileName)
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new ModelInfoDialog(model, profileName);
        await win.ShowDialog(owner);
    }

    public static async Task ShowCodePreviewAsync(string code, string? language)
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new CodePreviewDialog(code, language);
        await win.ShowDialog(owner);
    }

    /// <summary>
    /// Open-file picker for a conversation-bundle JSON. Returns the path
    /// the user selected, or null if cancelled.
    /// </summary>
    public static async Task<string?> PickImportFileAsync()
    {
        var owner = Owner;
        if (owner is null) return null;
        var result = await owner.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Import conversations",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Conversation bundle (.json)") { Patterns = new[] { "*.json" } },
                FilePickerFileTypes.All,
            },
        });
        return result.Count > 0 ? result[0].TryGetLocalPath() : null;
    }

    /// <summary>Save-file picker for exporting the conversation bundle.</summary>
    public static async Task<string?> PickExportFileAsync()
    {
        var owner = Owner;
        if (owner is null) return null;
        var result = await owner.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Export conversations",
            SuggestedFileName = $"llamachat-conversations-{System.DateTime.Now:yyyy-MM-dd}.json",
            DefaultExtension = "json",
            FileTypeChoices = new[]
            {
                new FilePickerFileType("Conversation bundle (.json)") { Patterns = new[] { "*.json" } },
            },
        });
        return result?.TryGetLocalPath();
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
