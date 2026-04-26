using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace LlamaCpp.Bindings.GGUFSuite.Services;

/// <summary>
/// Fetches and caches the HellaSwag validation dataset
/// (<c>hellaswag_val_full.txt</c>, ~10042 tasks) — the canonical
/// dataset llama.cpp's perplexity tool uses for its
/// <c>--hellaswag</c> mode. Mirrors the
/// <see cref="WikitextCorpus"/> caching pattern so first-time use
/// downloads once and subsequent runs reuse the cache.
/// </summary>
/// <remarks>
/// Source is the
/// <c>klosax/hellaswag_text_data</c> mirror referenced by upstream's
/// <c>scripts/get-hellaswag.sh</c>. The text format is documented in
/// <see cref="LlamaHellaswag.ParseUpstreamText"/>.
/// </remarks>
public static class HellaswagDataset
{
    /// <summary>Pinned upstream URL.</summary>
    public const string Url =
        "https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt";

    /// <summary>Cached path; shared with other test-data caches.</summary>
    public static string CachedPath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "llama-test-models", "hellaswag_val_full.txt");

    public static bool IsCached() => File.Exists(CachedPath());

    /// <summary>
    /// Download (if needed) and return the path to the cached
    /// HellaSwag validation file.
    /// </summary>
    public static async Task<string> EnsureAsync(
        IProgress<(long downloaded, long? total)>? downloadProgress = null,
        CancellationToken cancellationToken = default)
    {
        var dst = CachedPath();
        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
        if (File.Exists(dst)) return dst;

        var tmp = dst + ".tmp";
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
            using var response = await client.GetAsync(
                Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();
            var total = response.Content.Headers.ContentLength;
            await using (var src = await response.Content.ReadAsStreamAsync(cancellationToken))
            await using (var dstFs = File.Create(tmp))
            {
                var buf = new byte[64 * 1024];
                long downloaded = 0;
                int n;
                while ((n = await src.ReadAsync(buf.AsMemory(), cancellationToken)) > 0)
                {
                    await dstFs.WriteAsync(buf.AsMemory(0, n), cancellationToken);
                    downloaded += n;
                    downloadProgress?.Report((downloaded, total));
                }
            }
            File.Move(tmp, dst, overwrite: true);
            return dst;
        }
        catch
        {
            if (File.Exists(tmp))
            {
                try { File.Delete(tmp); } catch { /* best-effort */ }
            }
            throw;
        }
    }
}
