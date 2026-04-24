using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Avalonia;
using Avalonia.Platform;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Loads <c>Assets/sampling-profiles.json</c> once per process and matches
/// a model's GGUF metadata to a publisher-recommended sampler recipe. The
/// DB is a maintained lookup — we do not attempt to derive recommendations
/// from the model bytes themselves.
/// </summary>
/// <remarks>
/// First match wins. Matching is case-insensitive across both the
/// architecture string (exact equality) and the name pattern (regex). An
/// entry with an omitted key wildcards that key. If nothing matches, the
/// file's <c>fallback</c> block is returned; that block is guaranteed to
/// exist by the schema so callers never see a null.
/// </remarks>
public static class SamplingProfileDb
{
    private static readonly Lazy<SamplingProfileDatabase> _prodDb = new(LoadFromAsset);

    /// <summary>
    /// Look up the best-matching sampling profile for the given GGUF
    /// metadata using the embedded production DB. Returns the generic
    /// fallback if no rule matches.
    /// </summary>
    public static SamplingProfileEntry Match(string? architecture, string? modelName)
        => Match(_prodDb.Value, architecture, modelName);

    /// <summary>
    /// Testable overload — look up inside a caller-supplied database. Lets
    /// tests exercise the match logic against bespoke JSON without
    /// standing up the Avalonia resource system.
    /// </summary>
    public static SamplingProfileEntry Match(
        SamplingProfileDatabase db, string? architecture, string? modelName)
    {
        ArgumentNullException.ThrowIfNull(db);
        var arch = architecture?.Trim() ?? string.Empty;
        var name = modelName ?? string.Empty;

        foreach (var profile in db.Profiles)
        {
            var m = profile.Match;
            if (!string.IsNullOrEmpty(m?.Architecture) &&
                !string.Equals(m.Architecture, arch, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            if (!string.IsNullOrEmpty(m?.NamePattern))
            {
                try
                {
                    if (!Regex.IsMatch(name, m.NamePattern))
                        continue;
                }
                catch (ArgumentException)
                {
                    // Malformed regex in the DB — skip rather than blow up
                    // the whole auto-configure. Treat as non-match.
                    continue;
                }
            }
            return profile;
        }

        return db.Fallback;
    }

    /// <summary>
    /// Parse a sampling-profiles JSON document. Throws
    /// <see cref="InvalidDataException"/> if the document is malformed or
    /// missing the required <c>fallback</c> block.
    /// </summary>
    public static SamplingProfileDatabase Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var opts = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };

        var raw = JsonSerializer.Deserialize<RawDatabase>(json, opts)
            ?? throw new InvalidDataException("sampling-profiles JSON failed to parse.");

        if (raw.Fallback is null)
        {
            throw new InvalidDataException("sampling-profiles JSON is missing the required 'fallback' block.");
        }

        return new SamplingProfileDatabase(
            raw.Profiles ?? Array.Empty<SamplingProfileEntry>(),
            raw.Fallback);
    }

    private static SamplingProfileDatabase LoadFromAsset()
    {
        using var stream = AssetLoader.Open(
            new Uri("avares://LlamaCpp.Bindings.LlamaChat/Assets/sampling-profiles.json"));
        using var reader = new StreamReader(stream);
        return Parse(reader.ReadToEnd());
    }

    private sealed record RawDatabase(
        SamplingProfileEntry[]? Profiles,
        SamplingProfileEntry? Fallback);
}

public sealed record SamplingProfileDatabase(
    IReadOnlyList<SamplingProfileEntry> Profiles,
    SamplingProfileEntry Fallback);

public sealed record SamplingProfileEntry
{
    public string Id { get; init; } = string.Empty;
    public SamplingProfileMatch? Match { get; init; }
    public SamplingProfileValues Sampling { get; init; } = new();
    public string? Notes { get; init; }
    public string? Source { get; init; }
}

public sealed record SamplingProfileMatch
{
    public string? Architecture { get; init; }
    [JsonPropertyName("namePattern")]
    public string? NamePattern { get; init; }
}

public sealed record SamplingProfileValues
{
    public float? Temperature { get; init; }
    [JsonPropertyName("topP")]
    public float? TopP { get; init; }
    [JsonPropertyName("topK")]
    public int? TopK { get; init; }
    [JsonPropertyName("minP")]
    public float? MinP { get; init; }
    [JsonPropertyName("penaltyRepeat")]
    public float? PenaltyRepeat { get; init; }
}
