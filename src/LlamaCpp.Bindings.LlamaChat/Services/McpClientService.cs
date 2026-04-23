using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Application-wide facade owning the set of <see cref="McpClient"/> instances
/// and their cached tool/prompt/resource lists. One singleton — the MainWindow
/// VM, the Settings tab, and the chat-generation loop all read from here.
///
/// Responsibilities:
/// - Translate user CRUD on <see cref="McpServerConfig"/> entries into
///   connect/disconnect actions on the underlying clients.
/// - Expose connection state, errors, and capability metadata as observable
///   fields (<see cref="ServerEntry"/>) the UI can bind to.
/// - Fan out tool calls from the chat loop to the right client by tool-name
///   prefix.
/// - Maintain a ring-buffer of <see cref="McpExecutionLogEntry"/> for the log
///   dialog.
/// </summary>
public sealed class McpClientService : IDisposable
{
    public static McpClientService Instance { get; } = new();

    private McpClientService() { }

    public ObservableCollection<McpServerEntry> Servers { get; } = new();

    /// <summary>
    /// Ring buffer of request/response log entries, newest-first. Bounded to
    /// <see cref="LogCap"/> entries.
    /// </summary>
    public ObservableCollection<McpExecutionLogEntry> Log { get; } = new();
    private const int LogCap = 500;

    /// <summary>
    /// Fired whenever the Servers collection's state changes in a way that
    /// could affect which tools are advertised (connect, disconnect, reload).
    /// UI layers subscribe to refresh tool exposure.
    /// </summary>
    public event EventHandler? StateChanged;

    /// <summary>
    /// Bootstrap from <see cref="McpServerStore"/>. Idempotent — additional
    /// calls replace the current set.
    /// </summary>
    public async Task LoadAndConnectAsync(CancellationToken ct = default)
    {
        await DisconnectAllAsync().ConfigureAwait(false);
        var cfgs = McpServerStore.Load();
        foreach (var cfg in cfgs)
        {
            // Preserve the Guid round-trip so UI references stay stable across reloads.
            var entry = new McpServerEntry(cfg);
            Servers.Add(entry);
            _ = Task.Run(() => ConnectAsync(entry, ct), ct);
        }
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    /// Persist the current <see cref="Servers"/> list to disk. Call this after
    /// the user mutates the list from the settings UI.
    /// </summary>
    public void SaveToDisk()
    {
        try
        {
            McpServerStore.Save(Servers.Select(s => s.Config));
        }
        catch
        {
            // Non-fatal — the in-memory list is still the source of truth
            // for this process.
        }
    }

    public async Task AddServerAsync(McpServerConfig cfg)
    {
        var entry = new McpServerEntry(cfg);
        Servers.Add(entry);
        SaveToDisk();
        StateChanged?.Invoke(this, EventArgs.Empty);
        if (cfg.Enabled)
        {
            await ConnectAsync(entry).ConfigureAwait(false);
        }
    }

    public async Task UpdateServerAsync(McpServerEntry entry)
    {
        // Config mutated in place by the UI. Reconnect from scratch to pick up
        // URL / header / enable-state changes.
        await DisconnectAsync(entry).ConfigureAwait(false);
        SaveToDisk();
        StateChanged?.Invoke(this, EventArgs.Empty);
        if (entry.Config.Enabled)
        {
            await ConnectAsync(entry).ConfigureAwait(false);
        }
    }

    public async Task DeleteServerAsync(McpServerEntry entry)
    {
        await DisconnectAsync(entry).ConfigureAwait(false);
        Servers.Remove(entry);
        SaveToDisk();
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public async Task ToggleEnabledAsync(McpServerEntry entry)
    {
        entry.Config.Enabled = !entry.Config.Enabled;
        SaveToDisk();
        if (entry.Config.Enabled)
        {
            await ConnectAsync(entry).ConfigureAwait(false);
        }
        else
        {
            await DisconnectAsync(entry).ConfigureAwait(false);
            entry.State = McpConnectionState.Disabled;
        }
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public async Task ReconnectAsync(McpServerEntry entry)
    {
        await DisconnectAsync(entry).ConfigureAwait(false);
        await ConnectAsync(entry).ConfigureAwait(false);
    }

    private async Task ConnectAsync(McpServerEntry entry, CancellationToken ct = default)
    {
        if (!entry.Config.Enabled)
        {
            entry.State = McpConnectionState.Disabled;
            return;
        }
        if (string.IsNullOrWhiteSpace(entry.Config.Url))
        {
            entry.State = McpConnectionState.Error;
            entry.Error = "URL is empty.";
            return;
        }

        entry.State = McpConnectionState.Connecting;
        entry.Error = null;
        StateChanged?.Invoke(this, EventArgs.Empty);

        var handler = new HttpClientHandler();
        var http = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(60),
        };
        var client = new McpClient(entry.Config, http, LogAppend);
        try
        {
            await client.InitializeAsync(ct).ConfigureAwait(false);
            entry.Client = client;
            entry.Capabilities = client.Capabilities;

            if (client.Capabilities?.Tools == true)
            {
                entry.Tools = await client.ListToolsAsync(ct).ConfigureAwait(false);
            }
            if (client.Capabilities?.Prompts == true)
            {
                entry.Prompts = await client.ListPromptsAsync(ct).ConfigureAwait(false);
            }
            if (client.Capabilities?.Resources == true)
            {
                entry.Resources = await client.ListResourcesAsync(ct).ConfigureAwait(false);
            }

            entry.State = McpConnectionState.Ready;
        }
        catch (Exception ex)
        {
            entry.State = McpConnectionState.Error;
            entry.Error = ex.Message;
            client.Dispose();
        }
        finally
        {
            StateChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    private Task DisconnectAsync(McpServerEntry entry)
    {
        entry.Client?.Dispose();
        entry.Client = null;
        entry.Tools = Array.Empty<McpToolInfo>();
        entry.Prompts = Array.Empty<McpPromptInfo>();
        entry.Resources = Array.Empty<McpResourceInfo>();
        entry.Capabilities = null;
        return Task.CompletedTask;
    }

    public async Task DisconnectAllAsync()
    {
        foreach (var s in Servers.ToList())
        {
            await DisconnectAsync(s).ConfigureAwait(false);
        }
        Servers.Clear();
    }

    /// <summary>
    /// Enumerate all tools currently advertised by enabled, ready servers.
    /// Names are prefixed with the server name (<c>serverName__tool</c>) so
    /// the chat loop can dispatch a call back to the right server.
    /// </summary>
    public IEnumerable<(McpServerEntry Server, McpToolInfo Tool, string PrefixedName)> EnumerateTools()
    {
        foreach (var s in Servers)
        {
            if (s.State != McpConnectionState.Ready) continue;
            foreach (var t in s.Tools)
            {
                yield return (s, t, MakePrefixedName(s, t.Name));
            }
        }
    }

    /// <summary>
    /// Build the <c>tools</c> value the Jinja chat template consumes. Each
    /// entry is the OpenAI-style <c>{ type: "function", function: { name,
    /// description, parameters } }</c> shape that Qwen/DeepSeek/Hermes
    /// tool-use templates expect. Returns null if no ready server advertises
    /// tools so the template's non-tool branch renders.
    /// </summary>
    public IReadOnlyList<object?>? BuildToolsForTemplate()
    {
        var list = new List<object?>();
        foreach (var (server, tool, prefixed) in EnumerateTools())
        {
            // JsonElement → deserialised plain-object tree so the Jinja
            // interpreter can walk it (it expects Dictionary/List/primitives).
            object? parameters = null;
            try
            {
                parameters = JsonElementToPlain(tool.InputSchema);
            }
            catch
            {
                parameters = new Dictionary<string, object?>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object?>(),
                };
            }

            list.Add(new Dictionary<string, object?>
            {
                ["type"] = "function",
                ["function"] = new Dictionary<string, object?>
                {
                    ["name"] = prefixed,
                    ["description"] = tool.Description ?? string.Empty,
                    ["parameters"] = parameters,
                },
            });
        }
        return list.Count == 0 ? null : list;
    }

    private static object? JsonElementToPlain(JsonElement el)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.Object:
                var obj = new Dictionary<string, object?>();
                foreach (var p in el.EnumerateObject())
                {
                    obj[p.Name] = JsonElementToPlain(p.Value);
                }
                return obj;
            case JsonValueKind.Array:
                var arr = new List<object?>();
                foreach (var v in el.EnumerateArray())
                {
                    arr.Add(JsonElementToPlain(v));
                }
                return arr;
            case JsonValueKind.String:
                return el.GetString();
            case JsonValueKind.Number:
                return el.TryGetInt64(out var i) ? (object)i : el.GetDouble();
            case JsonValueKind.True:
                return true;
            case JsonValueKind.False:
                return false;
            case JsonValueKind.Null:
            case JsonValueKind.Undefined:
            default:
                return null;
        }
    }

    public static string MakePrefixedName(McpServerEntry server, string toolName)
    {
        var prefix = NameToPrefix(server.Config.Name);
        return string.IsNullOrEmpty(prefix) ? toolName : $"{prefix}__{toolName}";
    }

    /// <summary>
    /// Reverse the prefix convention from <see cref="MakePrefixedName"/>.
    /// Falls back to trying the raw name against every server if there's no
    /// prefix separator, so unprefixed tool calls from the model still work.
    /// </summary>
    public (McpServerEntry Server, string ToolName)? ResolveToolCall(string prefixedName)
    {
        var sep = prefixedName.IndexOf("__", StringComparison.Ordinal);
        if (sep > 0)
        {
            var prefix = prefixedName[..sep];
            var bare = prefixedName[(sep + 2)..];
            foreach (var s in Servers)
            {
                if (s.State != McpConnectionState.Ready) continue;
                if (NameToPrefix(s.Config.Name) == prefix
                    && s.Tools.Any(t => t.Name == bare))
                {
                    return (s, bare);
                }
            }
        }
        // Fallback: unprefixed — first server that advertises a tool of that name.
        foreach (var s in Servers)
        {
            if (s.State != McpConnectionState.Ready) continue;
            if (s.Tools.Any(t => t.Name == prefixedName))
            {
                return (s, prefixedName);
            }
        }
        return null;
    }

    private static string NameToPrefix(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return string.Empty;
        var sb = new System.Text.StringBuilder(name.Length);
        foreach (var ch in name)
        {
            if (char.IsLetterOrDigit(ch) || ch == '_') sb.Append(ch);
        }
        return sb.ToString();
    }

    public async Task<JsonElement> CallToolAsync(
        McpServerEntry server, string toolName, JsonElement arguments,
        CancellationToken ct = default)
    {
        if (server.Client is null)
        {
            throw new InvalidOperationException($"MCP server '{server.Config.Name}' is not connected.");
        }
        return await server.Client.CallToolAsync(toolName, arguments, ct).ConfigureAwait(false);
    }

    public async Task<string> GetPromptAsync(
        McpServerEntry server, string name,
        IReadOnlyDictionary<string, string>? args,
        CancellationToken ct = default)
    {
        if (server.Client is null)
        {
            throw new InvalidOperationException($"MCP server '{server.Config.Name}' is not connected.");
        }
        return await server.Client.GetPromptAsync(name, args, ct).ConfigureAwait(false);
    }

    public async Task<string> ReadResourceAsync(
        McpServerEntry server, string uri, CancellationToken ct = default)
    {
        if (server.Client is null)
        {
            throw new InvalidOperationException($"MCP server '{server.Config.Name}' is not connected.");
        }
        return await server.Client.ReadResourceAsync(uri, ct).ConfigureAwait(false);
    }

    private void LogAppend(McpExecutionLogEntry entry)
    {
        // Log events fire from background threads; marshal to UI.
        Dispatcher.UIThread.Post(() =>
        {
            Log.Insert(0, entry);
            while (Log.Count > LogCap) Log.RemoveAt(Log.Count - 1);
        });
    }

    public void Dispose()
    {
        foreach (var s in Servers)
        {
            s.Client?.Dispose();
        }
        Servers.Clear();
    }
}

/// <summary>
/// Per-server state bag combining the <see cref="McpServerConfig"/> (user
/// input) with the live connection state, capabilities, and cached lists.
/// Observable so the settings UI can react to connect/disconnect without a
/// separate ViewModel layer.
/// </summary>
public sealed class McpServerEntry : CommunityToolkit.Mvvm.ComponentModel.ObservableObject
{
    public McpServerConfig Config { get; }

    public McpServerEntry(McpServerConfig cfg) { Config = cfg; }

    internal McpClient? Client { get; set; }

    private McpConnectionState _state = McpConnectionState.Idle;
    public McpConnectionState State
    {
        get => _state;
        set
        {
            if (SetProperty(ref _state, value))
            {
                OnPropertyChanged(nameof(IsReady));
                OnPropertyChanged(nameof(StateLabel));
            }
        }
    }

    private string? _error;
    public string? Error
    {
        get => _error;
        set => SetProperty(ref _error, value);
    }

    private McpCapabilities? _capabilities;
    public McpCapabilities? Capabilities
    {
        get => _capabilities;
        set
        {
            if (SetProperty(ref _capabilities, value))
            {
                OnPropertyChanged(nameof(HasTools));
                OnPropertyChanged(nameof(HasPrompts));
                OnPropertyChanged(nameof(HasResources));
            }
        }
    }

    private IReadOnlyList<McpToolInfo> _tools = Array.Empty<McpToolInfo>();
    public IReadOnlyList<McpToolInfo> Tools
    {
        get => _tools;
        set
        {
            if (SetProperty(ref _tools, value)) OnPropertyChanged(nameof(ToolsSummary));
        }
    }

    private IReadOnlyList<McpPromptInfo> _prompts = Array.Empty<McpPromptInfo>();
    public IReadOnlyList<McpPromptInfo> Prompts
    {
        get => _prompts;
        set => SetProperty(ref _prompts, value);
    }

    private IReadOnlyList<McpResourceInfo> _resources = Array.Empty<McpResourceInfo>();
    public IReadOnlyList<McpResourceInfo> Resources
    {
        get => _resources;
        set => SetProperty(ref _resources, value);
    }

    public bool IsReady => State == McpConnectionState.Ready;
    public bool HasTools => Capabilities?.Tools == true;
    public bool HasPrompts => Capabilities?.Prompts == true;
    public bool HasResources => Capabilities?.Resources == true;

    public string StateLabel => State switch
    {
        McpConnectionState.Idle       => "Idle",
        McpConnectionState.Connecting => "Connecting…",
        McpConnectionState.Ready      => "Ready",
        McpConnectionState.Error      => "Error",
        McpConnectionState.Disabled   => "Disabled",
        _                             => State.ToString(),
    };

    public string ToolsSummary =>
        Tools.Count == 0 ? "no tools"
        : Tools.Count == 1 ? "1 tool"
        : $"{Tools.Count} tools";

    /// <summary>
    /// Call after mutating <see cref="Config"/> fields in place. The sidebar
    /// ListBox reads <c>Config.Name</c> / <c>Config.Url</c> through this
    /// entry, and <see cref="McpServerConfig"/> itself has no change
    /// notification, so we re-emit a generic property-changed so bindings
    /// that dot through <c>Config</c> reevaluate.
    /// </summary>
    public void OnConfigChanged() => OnPropertyChanged(nameof(Config));
}
