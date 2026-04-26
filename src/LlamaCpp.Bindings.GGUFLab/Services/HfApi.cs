using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace LlamaCpp.Bindings.GGUFLab.Services;

/// <summary>One repo entry returned by the HuggingFace search API.</summary>
public sealed record HfModelSummary(
    string Id,
    string? Author,
    DateTime? LastModified,
    long? Downloads,
    long? Likes,
    IReadOnlyList<string>? Tags,
    string? Pipeline,
    bool Gated)
{
    public string DisplayDownloads => Downloads is long d ? FormatCount(d) : "—";
    public string DisplayLikes     => Likes     is long l ? FormatCount(l) : "—";
    public string DisplayUpdated   => LastModified is DateTime dt
        ? RelativeAgo(dt)
        : "—";

    public string DisplayLibrary
    {
        get
        {
            if (Tags is null) return "";
            foreach (var t in Tags)
            {
                if (t == "gguf") return "GGUF";
                if (t == "safetensors") return "safetensors";
            }
            return "";
        }
    }

    private static string FormatCount(long n) =>
        n >= 1_000_000 ? $"{n / 1_000_000.0:F1}M"
      : n >= 1_000     ? $"{n / 1_000.0:F1}k"
      :                  n.ToString();

    private static string RelativeAgo(DateTime dt)
    {
        var span = DateTime.UtcNow - dt.ToUniversalTime();
        if (span.TotalDays   >= 365) return $"{(int)(span.TotalDays/365)}y ago";
        if (span.TotalDays   >= 30)  return $"{(int)(span.TotalDays/30)}mo ago";
        if (span.TotalDays   >= 1)   return $"{(int)span.TotalDays}d ago";
        if (span.TotalHours  >= 1)   return $"{(int)span.TotalHours}h ago";
        if (span.TotalMinutes>= 1)   return $"{(int)span.TotalMinutes}m ago";
        return "just now";
    }
}

/// <summary>One file in a repo's tree.</summary>
public sealed record HfModelFile(
    string Path,
    long? Size,
    string? Type,
    string? Oid)
{
    public bool IsDirectory => string.Equals(Type, "directory", StringComparison.OrdinalIgnoreCase);

    public string DisplaySize => Size switch
    {
        null         => "",
        < 1024       => $"{Size} B",
        < 1024 * 1024 => $"{Size.Value / 1024.0:F1} KB",
        < 1024L * 1024 * 1024 => $"{Size.Value / (1024.0 * 1024):F1} MB",
        _            => $"{Size.Value / (1024.0 * 1024 * 1024):F2} GB",
    };
}

/// <summary>Library filter for <see cref="HfApi.SearchAsync"/>.</summary>
public enum HfLibraryFilter { Any, Gguf, Safetensors }

/// <summary>Sort key for <see cref="HfApi.SearchAsync"/>.</summary>
public enum HfSortKey { Trending, Downloads, Likes, RecentlyUpdated }

/// <summary>
/// Thin client for the public HuggingFace Hub API. Search and tree
/// listing don't need a token; the optional <see cref="WorkspaceSettings.HuggingFaceToken"/>
/// is forwarded as <c>Authorization: Bearer …</c> so gated repos work.
/// </summary>
public sealed class HfApi : IDisposable
{
    private readonly HttpClient _http;
    private readonly Func<string?> _tokenProvider;

    public HfApi(Func<string?> tokenProvider)
    {
        _tokenProvider = tokenProvider;
        _http = new HttpClient
        {
            BaseAddress = new Uri("https://huggingface.co/"),
            Timeout = TimeSpan.FromSeconds(30),
        };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("LlamaCpp.Bindings.GGUFLab/1.0 (+https://github.com/christopherthompson81/LlamaCpp.Bindings)");
    }

    public void Dispose() => _http.Dispose();

    private void AttachAuth(HttpRequestMessage req)
    {
        var token = _tokenProvider();
        if (!string.IsNullOrWhiteSpace(token))
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
    }

    /// <summary>
    /// Search repositories. Empty <paramref name="query"/> still works —
    /// the endpoint then returns top results by the chosen <paramref name="sort"/>.
    /// </summary>
    public async Task<IReadOnlyList<HfModelSummary>> SearchAsync(
        string? query,
        HfLibraryFilter library,
        HfSortKey sort,
        int limit = 30,
        CancellationToken ct = default)
    {
        var qs = new List<string>();
        if (!string.IsNullOrWhiteSpace(query)) qs.Add($"search={Uri.EscapeDataString(query)}");
        switch (library)
        {
            case HfLibraryFilter.Gguf:        qs.Add("filter=gguf"); break;
            case HfLibraryFilter.Safetensors: qs.Add("filter=safetensors"); break;
        }
        qs.Add("sort=" + sort switch
        {
            HfSortKey.Downloads        => "downloads",
            HfSortKey.Likes            => "likes",
            HfSortKey.RecentlyUpdated  => "lastModified",
            _                          => "trendingScore", // default: Trending
        });
        qs.Add("direction=-1");
        qs.Add($"limit={limit}");
        qs.Add("full=true");  // include downloads/likes/tags

        var url = "api/models?" + string.Join("&", qs);
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        AttachAuth(req);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseContentRead, ct);
        resp.EnsureSuccessStatusCode();
        var payload = await resp.Content.ReadFromJsonAsync<List<HfApiModel>>(cancellationToken: ct)
                      ?? new List<HfApiModel>();
        var result = new List<HfModelSummary>(payload.Count);
        foreach (var m in payload)
        {
            result.Add(new HfModelSummary(
                Id:           m.Id,
                Author:       m.Author,
                LastModified: m.LastModified,
                Downloads:    m.Downloads,
                Likes:        m.Likes,
                Tags:         m.Tags,
                Pipeline:     m.PipelineTag,
                Gated:        m.Gated == true));
        }
        return result;
    }

    /// <summary>
    /// List files in a repo at the given <paramref name="revision"/>
    /// (default <c>main</c>). The tree endpoint returns LFS-aware sizes
    /// for big GGUFs; <c>siblings</c> on /api/models/{id} would not.
    /// </summary>
    public async Task<IReadOnlyList<HfModelFile>> ListFilesAsync(
        string repoId,
        string revision = "main",
        CancellationToken ct = default)
    {
        var url = $"api/models/{repoId}/tree/{revision}?recursive=true";
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        AttachAuth(req);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseContentRead, ct);
        resp.EnsureSuccessStatusCode();
        var payload = await resp.Content.ReadFromJsonAsync<List<HfApiTreeEntry>>(cancellationToken: ct)
                      ?? new List<HfApiTreeEntry>();
        var result = new List<HfModelFile>(payload.Count);
        foreach (var e in payload)
        {
            // Prefer LFS size when present (big binaries route through LFS).
            long? size = e.Size;
            if (size is null && e.Lfs is { } lfs && lfs.Size > 0) size = lfs.Size;
            result.Add(new HfModelFile(e.Path, size, e.Type, e.Oid));
        }
        return result;
    }

    /// <summary>
    /// Stream a file from the repo to <paramref name="destPath"/>. Reports
    /// <c>(downloaded, total)</c> bytes; <c>total</c> is null when the
    /// server doesn't include Content-Length (rare for HF; CDN usually
    /// surfaces it).
    /// </summary>
    public async Task DownloadAsync(
        string repoId,
        string filePath,
        string destPath,
        IProgress<(long Downloaded, long? Total)>? progress = null,
        CancellationToken ct = default,
        string revision = "main")
    {
        var url = $"{repoId}/resolve/{revision}/{filePath}";
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        AttachAuth(req);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        resp.EnsureSuccessStatusCode();

        long? total = resp.Content.Headers.ContentLength;
        Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);

        await using var src = await resp.Content.ReadAsStreamAsync(ct);
        // Atomic-ish: download to a sibling temp file then move into place.
        var tempPath = destPath + ".part";
        await using (var dst = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true))
        {
            var buf = new byte[1 << 16];
            long sum = 0;
            int n;
            while ((n = await src.ReadAsync(buf.AsMemory(0, buf.Length), ct)) > 0)
            {
                await dst.WriteAsync(buf.AsMemory(0, n), ct);
                sum += n;
                progress?.Report((sum, total));
            }
        }
        if (File.Exists(destPath)) File.Delete(destPath);
        File.Move(tempPath, destPath);
    }

    // --- API DTOs ---------------------------------------------------------

    private sealed class HfApiModel
    {
        [JsonPropertyName("id")]            public string Id { get; set; } = "";
        [JsonPropertyName("author")]        public string? Author { get; set; }
        [JsonPropertyName("lastModified")]  public DateTime? LastModified { get; set; }
        [JsonPropertyName("downloads")]     public long? Downloads { get; set; }
        [JsonPropertyName("likes")]         public long? Likes { get; set; }
        [JsonPropertyName("tags")]          public List<string>? Tags { get; set; }
        [JsonPropertyName("pipeline_tag")]  public string? PipelineTag { get; set; }
        [JsonPropertyName("gated")]         public bool? Gated { get; set; }
    }

    private sealed class HfApiTreeEntry
    {
        [JsonPropertyName("type")] public string? Type { get; set; }
        [JsonPropertyName("oid")]  public string? Oid { get; set; }
        [JsonPropertyName("size")] public long? Size { get; set; }
        [JsonPropertyName("path")] public string Path { get; set; } = "";
        [JsonPropertyName("lfs")]  public HfApiLfs? Lfs { get; set; }
    }

    private sealed class HfApiLfs
    {
        [JsonPropertyName("size")] public long Size { get; set; }
        [JsonPropertyName("oid")]  public string? Oid { get; set; }
    }
}
