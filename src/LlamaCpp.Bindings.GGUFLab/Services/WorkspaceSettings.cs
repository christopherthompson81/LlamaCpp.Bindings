using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings.GGUFLab.Services;

/// <summary>
/// Persistent app-level settings for GGUF Lab. Currently a single
/// <see cref="WorkspaceRoot"/> — the directory the HF Browser writes
/// downloads into and the Local Models page scans. Stored as JSON
/// alongside the OS app-data convention so multiple installs of the
/// app share the same workspace.
/// </summary>
public sealed class WorkspaceSettings
{
    public string WorkspaceRoot { get; set; } = DefaultWorkspaceRoot;

    /// <summary>HuggingFace API token for gated repos. Optional.</summary>
    public string? HuggingFaceToken { get; set; }

    public static string DefaultWorkspaceRoot =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-models");

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
                "GGUFLab");
            return Path.Combine(dir, "workspace-settings.json");
        }
    }

    public static WorkspaceSettings Load()
    {
        try
        {
            if (!File.Exists(StorePath)) return new WorkspaceSettings();
            var json = File.ReadAllText(StorePath);
            return JsonSerializer.Deserialize<WorkspaceSettings>(json, JsonOpts)
                ?? new WorkspaceSettings();
        }
        catch
        {
            // Corrupt JSON shouldn't crash the app — fall back to defaults
            // and the user can reset via Settings.
            return new WorkspaceSettings();
        }
    }

    public void Save()
    {
        var dir = Path.GetDirectoryName(StorePath)!;
        Directory.CreateDirectory(dir);
        File.WriteAllText(StorePath, JsonSerializer.Serialize(this, JsonOpts));
    }

    /// <summary>Ensure <see cref="WorkspaceRoot"/> exists on disk; safe to call repeatedly.</summary>
    public string EnsureWorkspaceRoot()
    {
        Directory.CreateDirectory(WorkspaceRoot);
        return WorkspaceRoot;
    }
}
