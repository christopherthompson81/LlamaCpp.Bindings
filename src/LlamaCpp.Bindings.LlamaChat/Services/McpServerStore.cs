using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// JSON-backed persistence for the user's <see cref="McpServerConfig"/>
/// list. Lives alongside <c>profiles.json</c> / <c>app-settings.json</c>.
/// Silent on corrupt files — returns an empty list so a bad JSON blob
/// doesn't brick startup.
/// </summary>
internal static class McpServerStore
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static string StorePath
    {
        get
        {
            var dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "LlamaChat");
            return Path.Combine(dir, "mcp-servers.json");
        }
    }

    public static IReadOnlyList<McpServerConfig> Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return Array.Empty<McpServerConfig>();
            var json = File.ReadAllText(StorePath);
            var loaded = JsonSerializer.Deserialize<List<McpServerConfig>>(json, JsonOpts);
            return loaded ?? new List<McpServerConfig>();
        }
        catch
        {
            return Array.Empty<McpServerConfig>();
        }
    }

    public static void Save(IEnumerable<McpServerConfig> servers)
    {
        var dir = Path.GetDirectoryName(StorePath)!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(servers, JsonOpts);
        File.WriteAllText(StorePath, json);
    }
}
