using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Reads and writes the user's list of <see cref="ModelProfile"/>s to a JSON
/// file under the platform's per-user config directory (Linux:
/// <c>$XDG_CONFIG_HOME/LlamaChat/profiles.json</c>; Windows:
/// <c>%APPDATA%\LlamaChat\profiles.json</c>). Missing file → a single seed
/// profile so the UI has something to show on first run.
/// </summary>
internal static class ProfileStore
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
            return Path.Combine(dir, "profiles.json");
        }
    }

    public static IReadOnlyList<ModelProfile> Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return Seed();
            var json = File.ReadAllText(StorePath);
            var loaded = JsonSerializer.Deserialize<List<ModelProfile>>(json, JsonOpts);
            if (loaded is null || loaded.Count == 0) return Seed();
            return loaded;
        }
        catch
        {
            // Corrupt file — fall back to seed rather than crash on launch.
            // The user can re-save from Settings to overwrite.
            return Seed();
        }
    }

    public static void Save(IEnumerable<ModelProfile> profiles)
    {
        var dir = Path.GetDirectoryName(StorePath)!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(profiles, JsonOpts);
        File.WriteAllText(StorePath, json);
    }

    private static IReadOnlyList<ModelProfile> Seed() => new[]
    {
        new ModelProfile { Name = "Default" },
    };
}
