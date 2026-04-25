using System.Security.Cryptography;
using System.Text;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Simple API-key authentication middleware. Not backed by ASP.NET Core's
/// full auth/authorization pipeline because the entire feature is "one
/// shared-secret header check" — pulling in identity/claims machinery
/// would be more scaffolding than logic.
/// </summary>
/// <remarks>
/// <para><b>Accepted header formats:</b></para>
/// <list type="bullet">
///   <item><c>Authorization: Bearer &lt;key&gt;</c> — OpenAI-compatible; what most client libraries send by default.</item>
///   <item><c>X-Api-Key: &lt;key&gt;</c> — fallback for environments where a Bearer header is inconvenient.</item>
/// </list>
///
/// <para><b>Bypass rules:</b> <c>/health</c> always passes through so
/// container liveness probes work without credentials. When no keys are
/// configured the middleware is a no-op — intentional for localhost dev
/// where auth just gets in the way.</para>
///
/// <para><b>Timing:</b> we run <see cref="CryptographicOperations.FixedTimeEquals"/>
/// per configured key so an attacker can't distinguish a near-match from
/// a total miss by response timing. Length mismatches still short-circuit
/// the comparison — the realistic risk model here (small set of shared
/// secrets, not a per-user password table) doesn't warrant the extra
/// hashing dance to paper that over.</para>
/// </remarks>
public static class ApiKeyAuth
{
    /// <summary>
    /// Load API keys from both the inline <see cref="Configuration.ServerOptions.ApiKeys"/>
    /// list and the optional <see cref="Configuration.ServerOptions.ApiKeyFile"/>
    /// path, returning a deduplicated byte-encoded set for the middleware
    /// to compare against.
    /// </summary>
    public static IReadOnlyList<byte[]> LoadKeys(
        IEnumerable<string> inline, string? filePath, ILogger? logger = null)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var k in inline)
        {
            if (!string.IsNullOrWhiteSpace(k))
            {
                seen.Add(k.Trim());
            }
        }
        if (!string.IsNullOrWhiteSpace(filePath))
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException(
                    $"LlamaServer:ApiKeyFile='{filePath}' does not exist.", filePath);
            }
            foreach (var raw in File.ReadAllLines(filePath))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                seen.Add(line);
            }
        }
        if (seen.Count > 0 && logger is not null)
        {
            logger.LogInformation("API key auth enabled with {Count} key(s)", seen.Count);
        }
        return seen.Select(Encoding.UTF8.GetBytes).ToArray();
    }

    /// <summary>
    /// Register the middleware on <paramref name="app"/>. Does nothing when
    /// <paramref name="validKeys"/> is empty — auth disabled.
    /// </summary>
    public static IApplicationBuilder UseApiKeyAuth(
        this IApplicationBuilder app, IReadOnlyList<byte[]> validKeys)
    {
        if (validKeys.Count == 0)
        {
            return app;
        }

        return app.Use(async (context, next) =>
        {
            // Liveness probes: always admitted.
            if (context.Request.Path.Equals("/health", StringComparison.OrdinalIgnoreCase))
            {
                await next();
                return;
            }

            var presented = ExtractKey(context.Request);
            if (presented is not null && Matches(presented, validKeys))
            {
                await next();
                return;
            }

            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            context.Response.ContentType = "application/json";
            // OpenAI-shaped error so client libraries surface a useful message.
            await context.Response.WriteAsync(
                "{\"error\":{\"message\":\"Invalid or missing API key.\"," +
                "\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}");
        });
    }

    private static string? ExtractKey(HttpRequest req)
    {
        if (req.Headers.TryGetValue("Authorization", out var auth))
        {
            var s = auth.ToString();
            if (s.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                return s["Bearer ".Length..].Trim();
            }
        }
        if (req.Headers.TryGetValue("X-Api-Key", out var xkey))
        {
            var s = xkey.ToString().Trim();
            if (s.Length > 0) return s;
        }
        return null;
    }

    private static bool Matches(string presented, IReadOnlyList<byte[]> validKeys)
    {
        var presentedBytes = Encoding.UTF8.GetBytes(presented);
        bool hit = false;
        foreach (var k in validKeys)
        {
            // FixedTimeEquals short-circuits on length mismatch before the
            // constant-time compare, so it's not fully length-blind — but
            // per-key iteration still masks which specific key matched, and
            // we don't OR-short-circuit here so every attempt runs through
            // every configured key.
            if (CryptographicOperations.FixedTimeEquals(presentedBytes, k))
            {
                hit = true;
            }
        }
        return hit;
    }
}
