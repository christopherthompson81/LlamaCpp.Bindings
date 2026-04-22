using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

internal static class AppSettingsStore
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
    };

    public static string StorePath
    {
        get
        {
            var dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "LlamaChat");
            return Path.Combine(dir, "app-settings.json");
        }
    }

    public static AppSettings Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return new AppSettings();
            var json = File.ReadAllText(StorePath);
            return JsonSerializer.Deserialize<AppSettings>(json, JsonOpts) ?? new AppSettings();
        }
        catch
        {
            return new AppSettings();
        }
    }

    public static void Save(AppSettings settings)
    {
        var dir = Path.GetDirectoryName(StorePath)!;
        Directory.CreateDirectory(dir);
        File.WriteAllText(StorePath, JsonSerializer.Serialize(settings, JsonOpts));
    }
}
