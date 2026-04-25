using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// JSON-backed persistence for <see cref="LocalServerConfig"/>. Mirrors
/// <see cref="AppSettingsStore"/> / <see cref="McpServerStore"/>: silent on
/// corrupt files (returns defaults) so a bad blob doesn't brick startup.
/// </summary>
internal static class LocalServerConfigStore
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
            return Path.Combine(dir, "local-server-config.json");
        }
    }

    public static LocalServerConfig Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return new LocalServerConfig();
            var json = File.ReadAllText(StorePath);
            return JsonSerializer.Deserialize<LocalServerConfig>(json, JsonOpts) ?? new LocalServerConfig();
        }
        catch
        {
            return new LocalServerConfig();
        }
    }

    public static void Save(LocalServerConfig config)
    {
        var dir = Path.GetDirectoryName(StorePath)!;
        Directory.CreateDirectory(dir);
        File.WriteAllText(StorePath, JsonSerializer.Serialize(config, JsonOpts));
    }
}
