using System.Collections.Concurrent;
using System.Globalization;
using System.Text;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// In-process counters + gauges for the <c>/metrics</c> endpoint.
/// Prometheus text-exposition format. Singleton; accessed concurrently
/// so every write goes through <see cref="Interlocked"/> or a
/// thread-safe container.
/// </summary>
/// <remarks>
/// V1 is deliberately minimal. Histograms / time-to-first-token
/// distributions are an add-on; counters and slot gauges are enough to
/// answer "how much traffic is this server serving" and "is the pool
/// saturated" at a glance.
/// </remarks>
public sealed class ServerMetrics
{
    // endpoint -> status code -> count
    private readonly ConcurrentDictionary<(string Endpoint, int Status), long> _requests = new();
    private long _tokensGenerated;
    private long _promptTokensIngested;
    private long _cachedTokensReused;

    public void IncrementRequest(string endpoint, int status)
    {
        _requests.AddOrUpdate((endpoint, status), 1, (_, v) => v + 1);
    }

    public void AddTokensGenerated(int n)
    {
        if (n > 0) Interlocked.Add(ref _tokensGenerated, n);
    }

    public void AddPromptTokensIngested(int n)
    {
        if (n > 0) Interlocked.Add(ref _promptTokensIngested, n);
    }

    public void AddCachedTokensReused(int n)
    {
        if (n > 0) Interlocked.Add(ref _cachedTokensReused, n);
    }

    /// <summary>
    /// Render the current counter + gauge state as Prometheus text. Slot
    /// gauges come from the live <see cref="SessionPool"/> snapshot so
    /// they're as-of-scrape, not last-request.
    /// </summary>
    public string Render(SessionPool pool)
    {
        var sb = new StringBuilder();

        sb.AppendLine("# HELP llama_requests_total Total HTTP requests by endpoint + status.");
        sb.AppendLine("# TYPE llama_requests_total counter");
        foreach (var kv in _requests)
        {
            var ((endpoint, status), count) = (kv.Key, kv.Value);
            sb.Append("llama_requests_total{endpoint=\"").Append(Escape(endpoint))
              .Append("\",status=\"").Append(status).Append("\"} ")
              .Append(count.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }

        sb.AppendLine("# HELP llama_tokens_generated_total Total generated (non-prompt) tokens across all requests.");
        sb.AppendLine("# TYPE llama_tokens_generated_total counter");
        sb.Append("llama_tokens_generated_total ")
          .Append(Volatile.Read(ref _tokensGenerated).ToString(CultureInfo.InvariantCulture)).Append('\n');

        sb.AppendLine("# HELP llama_tokens_prompt_total Total prompt tokens decoded (excludes cache hits).");
        sb.AppendLine("# TYPE llama_tokens_prompt_total counter");
        sb.Append("llama_tokens_prompt_total ")
          .Append(Volatile.Read(ref _promptTokensIngested).ToString(CultureInfo.InvariantCulture)).Append('\n');

        sb.AppendLine("# HELP llama_tokens_cached_total Total prompt tokens satisfied by the session-pool prefix cache.");
        sb.AppendLine("# TYPE llama_tokens_cached_total counter");
        sb.Append("llama_tokens_cached_total ")
          .Append(Volatile.Read(ref _cachedTokensReused).ToString(CultureInfo.InvariantCulture)).Append('\n');

        var slots = pool.Snapshot();
        sb.AppendLine("# HELP llama_slot_in_use 1 when the slot is currently leased, 0 otherwise.");
        sb.AppendLine("# TYPE llama_slot_in_use gauge");
        foreach (var s in slots)
        {
            sb.Append("llama_slot_in_use{slot_id=\"").Append(s.SlotIndex)
              .Append("\"} ").Append(s.InUse ? "1" : "0").Append('\n');
        }

        sb.AppendLine("# HELP llama_slot_cached_tokens Tokens currently in the slot's KV cache.");
        sb.AppendLine("# TYPE llama_slot_cached_tokens gauge");
        foreach (var s in slots)
        {
            sb.Append("llama_slot_cached_tokens{slot_id=\"").Append(s.SlotIndex)
              .Append("\"} ").Append(s.CachedTokenCount).Append('\n');
        }

        return sb.ToString();
    }

    /// <summary>Escape Prometheus label values (backslash, newline, quote).</summary>
    private static string Escape(string s) =>
        s.Replace("\\", "\\\\").Replace("\n", "\\n").Replace("\"", "\\\"");
}
