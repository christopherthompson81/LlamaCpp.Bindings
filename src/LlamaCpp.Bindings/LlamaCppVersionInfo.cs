namespace LlamaCpp.Bindings;

/// <summary>
/// Resolve the pinned llama.cpp version string at runtime. Used by the
/// investigation database to tag every measurement with the binary it
/// was taken against — when measurements taken months apart disagree,
/// the version stamp tells you whether to chase a real signal change
/// or an upstream change.
/// </summary>
/// <remarks>
/// The source of truth is <c>third_party/llama.cpp/VERSION</c> in the
/// repo root. In a development checkout it's reachable by walking up
/// from <see cref="AppContext.BaseDirectory"/>; in deployed contexts it
/// may be absent, in which case <see cref="GitDescribe"/> returns null.
/// </remarks>
public static class LlamaCppVersionInfo
{
    private static readonly Lazy<string?> _cached = new(Resolve);

    /// <summary><c>git_describe</c> string from the pinned VERSION file (e.g. <c>b8893-1-g86db42e97</c>), or null if unavailable.</summary>
    public static string? GitDescribe => _cached.Value;

    private static string? Resolve()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir))
        {
            var candidate = Path.Combine(dir, "third_party", "llama.cpp", "VERSION");
            if (File.Exists(candidate))
            {
                try
                {
                    foreach (var line in File.ReadAllLines(candidate))
                    {
                        var trimmed = line.Trim();
                        if (trimmed.StartsWith("git_describe", StringComparison.Ordinal))
                        {
                            // Format: "git_describe = b8893-1-g86db42e97"
                            var eq = trimmed.IndexOf('=');
                            if (eq > 0 && eq + 1 < trimmed.Length)
                                return trimmed[(eq + 1)..].Trim();
                        }
                    }
                }
                catch { /* fall through to null */ }
                return null;
            }
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }
}
