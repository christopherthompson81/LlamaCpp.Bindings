using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// JSON persistence for the user's conversation history. One file on disk;
/// loaded eagerly on startup, written whenever conversations change. Corrupt
/// file falls back to an empty list — losing history is bad but worse to
/// crash on launch.
/// </summary>
internal static class ConversationStore
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
            return Path.Combine(dir, "conversations.json");
        }
    }

    public static IReadOnlyList<Conversation> Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return Array.Empty<Conversation>();
            var json = File.ReadAllText(StorePath);
            var loaded = JsonSerializer.Deserialize<List<Conversation>>(json, JsonOpts);
            return loaded ?? (IReadOnlyList<Conversation>)Array.Empty<Conversation>();
        }
        catch
        {
            return Array.Empty<Conversation>();
        }
    }

    public static void Save(IEnumerable<Conversation> conversations)
    {
        var dir = Path.GetDirectoryName(StorePath)!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(conversations, JsonOpts);
        File.WriteAllText(StorePath, json);
    }

    public static void ExportToFile(IEnumerable<Conversation> conversations, string path)
    {
        File.WriteAllText(path, JsonSerializer.Serialize(conversations, JsonOpts));
    }

    public static IReadOnlyList<Conversation> ImportFromFile(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<List<Conversation>>(json, JsonOpts)
               ?? (IReadOnlyList<Conversation>)Array.Empty<Conversation>();
    }
}
