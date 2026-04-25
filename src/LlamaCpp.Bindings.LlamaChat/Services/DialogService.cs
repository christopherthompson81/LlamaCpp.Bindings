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

    public static async Task ShowAboutAsync()
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new AboutDialog();
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

    /// <summary>Show the MCP prompt picker modal. Returns the rendered prompt text, or null if cancelled.</summary>
    public static async Task<string?> ShowMcpPromptPickerAsync()
    {
        var owner = Owner;
        if (owner is null) return null;
        var win = new McpPromptPickerDialog();
        await win.ShowDialog(owner);
        return win.Result;
    }

    /// <summary>Show the MCP resource browser. Returns (uri, content) on attach, or null on cancel.</summary>
    public static async Task<(string Uri, string Content)?> ShowMcpResourceBrowserAsync()
    {
        var owner = Owner;
        if (owner is null) return null;
        var win = new McpResourcePickerDialog();
        await win.ShowDialog(owner);
        if (win.AttachedUri is null || win.AttachedContent is null) return null;
        return (win.AttachedUri, win.AttachedContent);
    }

    public static async Task ShowMcpExecutionLogAsync()
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new McpExecutionLogDialog();
        await win.ShowDialog(owner);
    }

    public static async Task ShowPendingAttachmentsAsync(MainWindowViewModel vm)
    {
        var owner = Owner;
        if (owner is null) return;
        var win = new PendingAttachmentsDialog(vm);
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

    /// <summary>
    /// Save-file picker for a single-conversation export via one of the
    /// <see cref="Exporters.IConversationExporter"/> implementations. The
    /// picker auto-populates the suggested filename from the conversation
    /// title and sets the file type filter to match the chosen exporter.
    /// </summary>
    public static async Task<string?> PickConversationExportFileAsync(
        Exporters.IConversationExporter exporter, string conversationTitle)
    {
        var owner = Owner;
        if (owner is null) return null;

        var safeTitle = System.Text.RegularExpressions.Regex.Replace(
            (conversationTitle ?? "conversation").Trim(), @"[^\w\-\. ]+", "_");
        if (string.IsNullOrWhiteSpace(safeTitle)) safeTitle = "conversation";
        var suggested = $"{safeTitle} — {System.DateTime.Now:yyyy-MM-dd}.{exporter.FileExtension}";

        var result = await owner.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = $"Export conversation — {exporter.DisplayName}",
            SuggestedFileName = suggested,
            DefaultExtension = exporter.FileExtension,
            FileTypeChoices = new[]
            {
                new FilePickerFileType(exporter.DisplayName) { Patterns = new[] { "*." + exporter.FileExtension } },
            },
        });
        return result?.TryGetLocalPath();
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

    /// <summary>
    /// Open-file picker for media attachments — images, audio, or both
    /// depending on what the caller's model consumes. Returns an array of
    /// local paths (empty on cancel). At least one of
    /// <paramref name="allowImages"/> or <paramref name="allowAudio"/>
    /// should be true; if both are false the picker returns empty without
    /// opening.
    /// </summary>
    public static async Task<IReadOnlyList<string>> PickMediaFilesAsync(
        bool allowImages = true, bool allowAudio = true)
    {
        var owner = Owner;
        if (owner is null) return System.Array.Empty<string>();
        if (!allowImages && !allowAudio) return System.Array.Empty<string>();

        var filters = new List<FilePickerFileType>();
        if (allowImages)
        {
            filters.Add(new FilePickerFileType("Images")
            {
                Patterns = new[] { "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp" },
                MimeTypes = new[] { "image/*" },
            });
        }
        if (allowAudio)
        {
            filters.Add(new FilePickerFileType("Audio")
            {
                Patterns = new[] { "*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a" },
                MimeTypes = new[] { "audio/*" },
            });
        }
        filters.Add(FilePickerFileTypes.All);

        var title = (allowImages, allowAudio) switch
        {
            (true, true)  => "Attach image or audio",
            (true, false) => "Attach images",
            (false, true) => "Attach audio",
            _             => "Attach",
        };

        var result = await owner.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = true,
            FileTypeFilter = filters.ToArray(),
        });
        var paths = new List<string>(result.Count);
        foreach (var file in result)
        {
            if (file.TryGetLocalPath() is { } p) paths.Add(p);
        }
        return paths;
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

    public static async Task<string?> PickServerExecutableAsync()
    {
        var owner = Owner;
        if (owner is null) return null;
        var result = await owner.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select LlamaCpp.Bindings.Server (.dll or .exe)",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Server executable")
                {
                    Patterns = new[] { "LlamaCpp.Bindings.Server.dll", "LlamaCpp.Bindings.Server.exe" },
                },
                FilePickerFileTypes.All,
            },
        });
        return result.Count > 0 ? result[0].TryGetLocalPath() : null;
    }
}
