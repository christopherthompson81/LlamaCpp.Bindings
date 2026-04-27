namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tests for <see cref="LlamaSensitivityProfile"/> JSONC I/O — schema
/// version handling, comment/trailing-comma tolerance, and round-trip
/// fidelity. The two committed reference profiles in
/// <c>data/profiles/</c> are loaded as-is to lock the on-disk schema
/// against accidental drift.
/// </summary>
public class SensitivityProfileTests
{
    private static string RepoRoot()
    {
        // Walk up from the test bin dir until we hit the repo root marker.
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "LlamaCpp.Bindings.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root.");
    }

    [Fact]
    public void CommittedQwen3Profiles_LoadAndExposeExpectedFields()
    {
        var profilesDir = Path.Combine(RepoRoot(), "data", "profiles");
        foreach (var name in new[] { "qwen3-0.6B.profile.json", "qwen3-1.7B.profile.json" })
        {
            var path = Path.Combine(profilesDir, name);
            Assert.True(File.Exists(path), $"missing reference profile: {path}");

            var p = LlamaSensitivityProfile.LoadFromJson(path);
            Assert.Equal(LlamaSensitivityProfile.CurrentSchemaVersion, p.SchemaVersion);
            Assert.Equal("qwen3", p.ArchitectureId);
            Assert.Equal(28, p.LayerCount);
            Assert.Equal("ablation", p.Provenance.Method);
            Assert.NotNull(p.Provenance.SourceParameterCount);
            Assert.True(p.Provenance.SourceParameterCount > 0);
            // Run 17 expanded profiles add output.weight + token_embd.weight
            // as proper categories on top of the original 7 (Run 16) — the
            // recipe builder needs per-tensor sensitivity for these to
            // avoid the over-protective UncategorizedProtections fallback.
            Assert.Equal(9, p.Categories.Count);
            // ffn_up is the most-sensitive Q4_K category on both reference
            // profiles — Run 9 / 13 finding. Lock it in.
            Assert.Equal("ffn_up", p.CategoriesByDescendingSensitivityAtQ4K.First());
        }
    }

    [Fact]
    public void RoundTrip_PreservesAllFields()
    {
        var src = Path.Combine(RepoRoot(), "data", "profiles", "qwen3-0.6B.profile.json");
        var loaded = LlamaSensitivityProfile.LoadFromJson(src);
        var tmp = Path.GetTempFileName();
        try
        {
            loaded.SaveToJson(tmp);
            var reloaded = LlamaSensitivityProfile.LoadFromJson(tmp);
            Assert.Equal(loaded.SchemaVersion, reloaded.SchemaVersion);
            Assert.Equal(loaded.ArchitectureId, reloaded.ArchitectureId);
            Assert.Equal(loaded.F16BaselinePerplexity, reloaded.F16BaselinePerplexity);
            Assert.Equal(loaded.Categories.Count, reloaded.Categories.Count);
            foreach (var (key, original) in loaded.Categories)
            {
                var rt = reloaded.Categories[key];
                Assert.Equal(original.RecommendedFloor, rt.RecommendedFloor);
                Assert.Equal(original.DeltaPplByType.Count, rt.DeltaPplByType.Count);
            }
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public void Loader_AcceptsCommentsAndTrailingCommas()
    {
        // Hand-authored JSONC: line comments, block comments, trailing
        // commas after both array entries and object members.
        var jsonc = """
            {
              // schema version is required
              "schemaVersion": 1,
              "architectureId": "test-arch",
              "layerCount": 4,
              "familyNotes": "synthetic for test",
              "provenance": {
                /* hand-authored profile */
                "method": "hand-authored",
                "sourceModel": null,
                "sourceParameterCount": null,
                "corpus": null,
                "builtAtUtc": null,
                "builderVersion": null,
              },
              "f16BaselinePerplexity": 10.0,
              "baselineContextSize": 512,
              "categories": {
                "ffn_up": {
                  "deltaPplByType": { "Q4_K": 0.5, "Q6_K": 0.05, },
                  "recommendedFloor": "Q4_K",
                  "notes": "highest leverage on every arch tested",
                },
              },
            }
            """;
        var tmp = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tmp, jsonc);
            var p = LlamaSensitivityProfile.LoadFromJson(tmp);
            Assert.Equal("hand-authored", p.Provenance.Method);
            Assert.Equal("test-arch", p.ArchitectureId);
            Assert.Single(p.Categories);
            Assert.Equal("highest leverage on every arch tested", p.Categories["ffn_up"].Notes);
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public void Loader_RejectsFutureSchemaVersion()
    {
        var json = """
            { "schemaVersion": 999, "architectureId": "x", "layerCount": 0,
              "familyNotes": null,
              "provenance": { "method": "ablation", "sourceModel": null,
                "sourceParameterCount": null, "corpus": null,
                "builtAtUtc": null, "builderVersion": null },
              "f16BaselinePerplexity": 0, "baselineContextSize": 0,
              "categories": {} }
            """;
        var tmp = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tmp, json);
            var ex = Assert.Throws<InvalidDataException>(() => LlamaSensitivityProfile.LoadFromJson(tmp));
            Assert.Contains("schema v999", ex.Message);
        }
        finally
        {
            File.Delete(tmp);
        }
    }
}
