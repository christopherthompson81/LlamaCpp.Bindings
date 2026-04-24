using LlamaCpp.Bindings.LlamaChat.Services;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Match logic for <see cref="SamplingProfileDb"/>. Uses hand-crafted JSON
/// instead of the production asset so tests run without an Avalonia
/// runtime, and so edits to the prod DB don't silently re-map test
/// expectations.
/// </summary>
public class SamplingProfileDbTests
{
    private const string Json = """
    {
      "profiles": [
        {
          "id": "qwen3-thinking",
          "match": { "architecture": "qwen3", "namePattern": "(?i)thinking|reasoner|qwq" },
          "sampling": { "temperature": 0.6, "topP": 0.95, "topK": 20, "minP": 0.0, "penaltyRepeat": 1.0 }
        },
        {
          "id": "qwen3",
          "match": { "architecture": "qwen3" },
          "sampling": { "temperature": 0.7, "topP": 0.8, "topK": 20, "minP": 0.0 }
        },
        {
          "id": "llama3",
          "match": { "architecture": "llama", "namePattern": "(?i)llama[\\s._-]?3" },
          "sampling": { "temperature": 0.6, "topP": 0.9 }
        },
        {
          "id": "broken-regex",
          "match": { "architecture": "mistral", "namePattern": "((" },
          "sampling": { "temperature": 0.1 }
        }
      ],
      "fallback": {
        "id": "generic",
        "sampling": { "temperature": 0.7, "topP": 0.9, "topK": 40, "minP": 0.05 }
      }
    }
    """;

    [Fact]
    public void Parse_Requires_Fallback()
    {
        Assert.Throws<System.IO.InvalidDataException>(() =>
            SamplingProfileDb.Parse("""{ "profiles": [] }"""));
    }

    [Fact]
    public void Match_Picks_First_Match_Wins()
    {
        var db = SamplingProfileDb.Parse(Json);

        // "Qwen3-0.6B-Thinking" should hit qwen3-thinking, not the generic qwen3 entry below it.
        var hit = SamplingProfileDb.Match(db, "qwen3", "Qwen3-0.6B-Thinking");
        Assert.Equal("qwen3-thinking", hit.Id);
    }

    [Fact]
    public void Match_Falls_Through_To_Family_When_Subvariant_Not_Matched()
    {
        var db = SamplingProfileDb.Parse(Json);
        var hit = SamplingProfileDb.Match(db, "qwen3", "Qwen3-0.6B-Instruct");
        Assert.Equal("qwen3", hit.Id);
    }

    [Fact]
    public void Match_Returns_Fallback_When_No_Rule_Matches()
    {
        var db = SamplingProfileDb.Parse(Json);
        var hit = SamplingProfileDb.Match(db, "gpt-oss", "GPT-OSS-20B");
        Assert.Equal("generic", hit.Id);
    }

    [Fact]
    public void Match_Is_Case_Insensitive_On_Architecture_And_Name()
    {
        var db = SamplingProfileDb.Parse(Json);
        var hit1 = SamplingProfileDb.Match(db, "QWEN3", "qwen3-thinking");
        Assert.Equal("qwen3-thinking", hit1.Id);

        var hit2 = SamplingProfileDb.Match(db, "qwen3", "QWEN3-INSTRUCT");
        Assert.Equal("qwen3", hit2.Id);
    }

    [Fact]
    public void Match_Regex_With_Whitespace_Separator()
    {
        var db = SamplingProfileDb.Parse(Json);
        // All three spellings should land on "llama3".
        Assert.Equal("llama3", SamplingProfileDb.Match(db, "llama", "Llama-3-8B-Instruct").Id);
        Assert.Equal("llama3", SamplingProfileDb.Match(db, "llama", "Llama 3.2 1B").Id);
        Assert.Equal("llama3", SamplingProfileDb.Match(db, "llama", "llama_3.1_70b").Id);

        // Llama-2 should NOT match (the regex rejects it).
        Assert.Equal("generic", SamplingProfileDb.Match(db, "llama", "Llama-2-7B").Id);
    }

    [Fact]
    public void Match_Tolerates_Malformed_Regex_In_Db()
    {
        var db = SamplingProfileDb.Parse(Json);
        // The "broken-regex" entry has namePattern="((" which can't compile —
        // it should be treated as a non-match, falling through to fallback.
        var hit = SamplingProfileDb.Match(db, "mistral", "Mistral-7B");
        Assert.Equal("generic", hit.Id);
    }

    [Fact]
    public void Match_Handles_Null_Inputs_Gracefully()
    {
        var db = SamplingProfileDb.Parse(Json);
        var hit = SamplingProfileDb.Match(db, null, null);
        Assert.Equal("generic", hit.Id);
    }

    /// <summary>
    /// Parse the shipped <c>Assets/sampling-profiles.json</c> as a
    /// regression gate — a typo or bad regex in the maintained DB should
    /// surface here rather than at runtime when the user clicks "Automagic".
    /// The file is located relative to the repo root (one level up from
    /// the test assembly's build output).
    /// </summary>
    [Fact]
    public void ProductionDatabase_Parses_And_Includes_Fallback()
    {
        // Walk up from the test assembly output to the repo root, then
        // descend into LlamaChat/Assets. Works both for "dotnet test" runs
        // (bin/Debug/netX/) and ad-hoc direct executions.
        var probe = AppContext.BaseDirectory;
        string? repoRoot = null;
        var d = new DirectoryInfo(probe);
        while (d is not null)
        {
            if (File.Exists(Path.Combine(d.FullName, "LlamaCpp.Bindings.slnx")))
            {
                repoRoot = d.FullName;
                break;
            }
            d = d.Parent;
        }
        Assert.NotNull(repoRoot);

        var path = Path.Combine(
            repoRoot!, "src", "LlamaCpp.Bindings.LlamaChat", "Assets", "sampling-profiles.json");
        Assert.True(File.Exists(path), $"Missing shipped DB at {path}");

        var db = SamplingProfileDb.Parse(File.ReadAllText(path));
        Assert.NotEmpty(db.Profiles);
        Assert.Equal("generic", db.Fallback.Id);

        // Spot-check a known entry so accidental renames show up here.
        Assert.Contains(db.Profiles, p => p.Id == "qwen3-thinking");
    }
}
