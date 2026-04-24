using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LlamaCpp.Bindings.LlamaChat.Models;
using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Drives the Tools/MCP tab in the Settings dialog. The tab has two sides:
/// a list of configured servers on the left, and an editor/form on the right
/// for the currently-selected server. Add/Delete/Connect/Toggle actions route
/// into <see cref="McpClientService"/> so persistence + connection state stay
/// in sync.
/// </summary>
public partial class McpSettingsViewModel : ObservableObject
{
    public ObservableCollection<McpServerEntry> Servers => McpClientService.Instance.Servers;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DeleteServerCommand), nameof(ReconnectServerCommand),
                                 nameof(ToggleEnabledCommand), nameof(SaveSelectedCommand))]
    private McpServerEntry? _selectedServer;

    [ObservableProperty] private string _draftName = string.Empty;
    [ObservableProperty] private string _draftUrl = string.Empty;
    [ObservableProperty] private string _draftHeaders = string.Empty;

    // Per-field validation state. Non-null message = invalid. The
    // IsXxxInvalid bools exist so the XAML can drive both the inline error
    // TextBlock and a Classes.invalid toggle on the TextBox from the same
    // signal.
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsNameInvalid), nameof(HasValidationErrors))]
    [NotifyCanExecuteChangedFor(nameof(SaveSelectedCommand), nameof(ReconnectServerCommand),
                                 nameof(ToggleEnabledCommand))]
    private string? _nameError;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsUrlInvalid), nameof(HasValidationErrors))]
    [NotifyCanExecuteChangedFor(nameof(SaveSelectedCommand), nameof(ReconnectServerCommand),
                                 nameof(ToggleEnabledCommand))]
    private string? _urlError;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsHeadersInvalid), nameof(HasValidationErrors))]
    [NotifyCanExecuteChangedFor(nameof(SaveSelectedCommand), nameof(ReconnectServerCommand),
                                 nameof(ToggleEnabledCommand))]
    private string? _headersError;

    public bool IsNameInvalid => !string.IsNullOrEmpty(NameError);
    public bool IsUrlInvalid => !string.IsNullOrEmpty(UrlError);
    public bool IsHeadersInvalid => !string.IsNullOrEmpty(HeadersError);
    public bool HasValidationErrors => IsNameInvalid || IsUrlInvalid || IsHeadersInvalid;

    /// <summary>Status line shown at the bottom of the tab (save/connection feedback).</summary>
    [ObservableProperty] private string _status = string.Empty;

    public McpSettingsViewModel()
    {
        if (Servers.Count == 0)
        {
            // Kick off the store load lazily — some callers instantiate this
            // VM before App.axaml's bootstrap has run.
            _ = McpClientService.Instance.LoadAndConnectAsync();
        }
        SelectedServer = Servers.FirstOrDefault();
    }

    partial void OnSelectedServerChanged(McpServerEntry? value)
    {
        if (value is null)
        {
            DraftName = string.Empty;
            DraftUrl = string.Empty;
            DraftHeaders = string.Empty;
            return;
        }
        DraftName = value.Config.Name;
        DraftUrl = value.Config.Url;
        DraftHeaders = string.Join("\n",
            value.Config.Headers.Select(kv => $"{kv.Key}: {kv.Value}"));
    }

    partial void OnDraftNameChanged(string value) =>
        NameError = string.IsNullOrWhiteSpace(value) ? "Name is required." : null;

    partial void OnDraftUrlChanged(string value) => UrlError = ValidateUrl(value);

    partial void OnDraftHeadersChanged(string value) => HeadersError = ValidateHeaders(value);

    private static string? ValidateUrl(string value)
    {
        var trimmed = value?.Trim() ?? string.Empty;
        if (trimmed.Length == 0) return "URL is required.";
        if (!Uri.TryCreate(trimmed, UriKind.Absolute, out var uri))
            return "Not a valid URL.";
        if (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
            return "URL must use http:// or https://.";
        return null;
    }

    private static string? ValidateHeaders(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return null;
        int lineNo = 0;
        foreach (var raw in value.Split('\n'))
        {
            lineNo++;
            var line = raw.Trim();
            if (line.Length == 0) continue;
            var colon = line.IndexOf(':');
            if (colon <= 0) return $"Line {lineNo}: expected \"Key: value\".";
            if (line[..colon].Trim().Length == 0)
                return $"Line {lineNo}: header key is empty.";
        }
        return null;
    }

    [RelayCommand]
    private async Task AddServerAsync()
    {
        var cfg = new McpServerConfig
        {
            Name = "New server",
            Url = string.Empty,
            Enabled = false,
        };
        await McpClientService.Instance.AddServerAsync(cfg);
        SelectedServer = Servers.LastOrDefault();
        Status = "Server added — fill in the URL and click Save.";
    }

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private async Task DeleteServerAsync()
    {
        if (SelectedServer is null) return;
        var target = SelectedServer;
        var name = string.IsNullOrWhiteSpace(target.Config.Name) ? "Untitled server" : target.Config.Name;
        var choice = await DialogService.ConfirmAsync(
            "Delete MCP server",
            $"Delete MCP server \"{name}\"? This closes the session and removes it from mcp-servers.json.",
            new[]
            {
                ("cancel", "Cancel", false, false),
                ("delete", "Delete", true, true),
            });
        if (choice != "delete") return;

        await McpClientService.Instance.DeleteServerAsync(target);
        SelectedServer = Servers.FirstOrDefault();
        Status = $"Deleted '{target.Config.Name}'.";
    }

    [RelayCommand(CanExecute = nameof(CanCommitDrafts))]
    private async Task SaveSelectedAsync()
    {
        if (SelectedServer is null) return;
        CommitDrafts(SelectedServer);
        await McpClientService.Instance.UpdateServerAsync(SelectedServer);
        Status = $"Saved '{SelectedServer.Config.Name}'.";
    }

    [RelayCommand(CanExecute = nameof(CanCommitDrafts))]
    private async Task ReconnectServerAsync()
    {
        if (SelectedServer is null) return;
        // Reconnect is usually pressed right after the user edited the URL —
        // commit drafts first so we don't reconnect with stale config.
        CommitDrafts(SelectedServer);
        Status = $"Reconnecting to '{SelectedServer.Config.Name}'…";
        await McpClientService.Instance.ReconnectAsync(SelectedServer);
        Status = SelectedServer.State switch
        {
            McpConnectionState.Ready => $"Connected: {SelectedServer.ToolsSummary}.",
            McpConnectionState.Error => $"Error: {SelectedServer.Error}",
            _ => $"State: {SelectedServer.StateLabel}",
        };
    }

    [RelayCommand(CanExecute = nameof(CanCommitDrafts))]
    private async Task ToggleEnabledAsync()
    {
        if (SelectedServer is null) return;
        // Same reason as Reconnect: "Connect on startup" is often ticked
        // right after filling in the URL. If we don't commit drafts first
        // the connect attempt runs against the stub (empty URL) and fails
        // with "URL is empty" while the textbox visibly shows a URL.
        CommitDrafts(SelectedServer);
        await McpClientService.Instance.ToggleEnabledAsync(SelectedServer);
        Status = SelectedServer.State switch
        {
            McpConnectionState.Ready    => $"Connected: {SelectedServer.ToolsSummary}.",
            McpConnectionState.Disabled => $"Disabled '{SelectedServer.Config.Name}'.",
            McpConnectionState.Error    => $"Error: {SelectedServer.Error}",
            _                           => $"State: {SelectedServer.StateLabel}",
        };
    }

    /// <summary>
    /// Copy the editor's Draft* fields into the entry's backing
    /// <see cref="McpServerConfig"/>. Called by every command that acts on
    /// the currently-selected server, so "type + click action" flows work
    /// without requiring an explicit Save beforehand.
    /// </summary>
    private void CommitDrafts(McpServerEntry entry)
    {
        entry.Config.Name = DraftName.Trim();
        entry.Config.Url = DraftUrl.Trim();
        entry.Config.Headers = ParseHeaders(DraftHeaders);
        // Nudge list-item rendering (sidebar shows Config.Name + Config.Url)
        // so the newly-typed name shows up without waiting for a selection
        // change. McpServerConfig itself doesn't notify — the entry wrapper
        // does, and the list-item template reads through it.
        entry.OnConfigChanged();
    }

    private bool HasSelection() => SelectedServer is not null;
    private bool CanCommitDrafts() => HasSelection() && !HasValidationErrors;

    /// <summary>
    /// Parse the multi-line "Key: value" textbox into a case-insensitive dict.
    /// Lines that don't contain a colon are skipped — lenient by design since
    /// this is free-form user input.
    /// </summary>
    private static System.Collections.Generic.Dictionary<string, string> ParseHeaders(string text)
    {
        var dict = new System.Collections.Generic.Dictionary<string, string>(
            StringComparer.OrdinalIgnoreCase);
        if (string.IsNullOrWhiteSpace(text)) return dict;
        foreach (var rawLine in text.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.Length == 0) continue;
            var colon = line.IndexOf(':');
            if (colon <= 0) continue;
            var key = line[..colon].Trim();
            var value = line[(colon + 1)..].Trim();
            if (key.Length == 0) continue;
            dict[key] = value;
        }
        return dict;
    }
}
