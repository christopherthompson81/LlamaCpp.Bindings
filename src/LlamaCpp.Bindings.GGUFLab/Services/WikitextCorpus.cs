using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Parquet;
using Parquet.Schema;

namespace LlamaCpp.Bindings.GGUFLab.Services;

/// <summary>
/// Fetches and caches the wikitext-2 raw test set — the de-facto corpus for
/// quoting llama.cpp perplexity numbers. Source is HuggingFace's parquet
/// build of <c>Salesforce/wikitext</c>; we decode the single <c>text</c>
/// column and concatenate rows to reproduce the original
/// <c>wiki.test.raw</c> byte-for-byte.
/// </summary>
/// <remarks>
/// <para>
/// Why this matters: every published "Q4_K_M PPL = 6.34" number in the
/// llama.cpp ecosystem is computed against this exact file with
/// <c>n_ctx=512</c> and second-half scoring. Without the same corpus,
/// our numbers are internally consistent but not directly comparable to
/// the published ones.
/// </para>
/// <para>
/// The download (~723 KB parquet, ~1.18 MB decoded) is cached to
/// <c>~/.cache/llama-test-models/wiki.test.raw</c> on first use and reused
/// across runs. The cache directory is intentionally shared with
/// <c>TestModelProvider</c> in the test project so a single machine cache
/// covers both.
/// </para>
/// </remarks>
public static class WikitextCorpus
{
    /// <summary>Pinned HF parquet URL for wikitext-2-raw-v1 test split.</summary>
    public const string TestParquetUrl =
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet";

    /// <summary>
    /// Download (if needed), decode, and cache the wikitext-2-raw test
    /// set. Returns the cached file path; callers can read it back via
    /// <see cref="File.ReadAllTextAsync(string, CancellationToken)"/> or
    /// just hand the path to a <c>FilePicker</c>-style consumer.
    /// </summary>
    /// <param name="downloadProgress">
    /// Optional sink for download bytes. <c>(downloaded, total?)</c> —
    /// total is null when the server doesn't send Content-Length. Only
    /// invoked during the parquet download; decode is fast and not
    /// progress-reported.
    /// </param>
    public static async Task<string> EnsureTestRawAsync(
        IProgress<(long downloaded, long? total)>? downloadProgress = null,
        CancellationToken cancellationToken = default)
    {
        var cacheDir = CacheDirectory();
        Directory.CreateDirectory(cacheDir);
        var decodedPath = Path.Combine(cacheDir, "wiki.test.raw");
        if (File.Exists(decodedPath))
        {
            return decodedPath;
        }

        // Download to a sibling .parquet so a partial decode doesn't
        // corrupt the cache. Move into place atomically at the end.
        var parquetPath = Path.Combine(cacheDir, "wiki.test.parquet");
        var tmpDecoded = decodedPath + ".tmp";

        try
        {
            await DownloadParquetAsync(parquetPath, downloadProgress, cancellationToken);
            await DecodeToPlainTextAsync(parquetPath, tmpDecoded, cancellationToken);
            File.Move(tmpDecoded, decodedPath, overwrite: false);
            return decodedPath;
        }
        catch
        {
            if (File.Exists(tmpDecoded))
            {
                try { File.Delete(tmpDecoded); } catch { /* best-effort */ }
            }
            throw;
        }
        finally
        {
            // Keep the parquet around so later schema changes / debugging
            // don't require a re-download. The decoded text is what
            // perplexity actually reads.
        }
    }

    /// <summary>
    /// True when the decoded test corpus is already on disk and
    /// <see cref="EnsureTestRawAsync"/> would return without network I/O.
    /// </summary>
    public static bool IsCached() =>
        File.Exists(Path.Combine(CacheDirectory(), "wiki.test.raw"));

    private static string CacheDirectory() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "llama-test-models");

    private static async Task DownloadParquetAsync(
        string destination,
        IProgress<(long downloaded, long? total)>? progress,
        CancellationToken cancellationToken)
    {
        // 5-minute timeout is generous for a ~700 KB file but covers slow
        // links. HttpClient is created per call — we don't fetch this
        // often enough to justify a long-lived client.
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        using var response = await client.GetAsync(
            TestParquetUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        var total = response.Content.Headers.ContentLength;
        var tmp = destination + ".part";
        try
        {
            await using (var src = await response.Content.ReadAsStreamAsync(cancellationToken))
            await using (var dst = File.Create(tmp))
            {
                var buf = new byte[64 * 1024];
                long downloaded = 0;
                int n;
                while ((n = await src.ReadAsync(buf.AsMemory(), cancellationToken)) > 0)
                {
                    await dst.WriteAsync(buf.AsMemory(0, n), cancellationToken);
                    downloaded += n;
                    progress?.Report((downloaded, total));
                }
            }
            File.Move(tmp, destination, overwrite: true);
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

    /// <summary>
    /// Read every row of the single <c>text</c> column from the parquet
    /// and concatenate. The HF parquet stores each wikitext "line" as one
    /// row with embedded newlines and no trailing separator, so a plain
    /// concat reproduces the original <c>wiki.test.raw</c> byte stream.
    /// </summary>
    private static async Task DecodeToPlainTextAsync(
        string parquetPath, string outputPath, CancellationToken cancellationToken)
    {
        await using var src = File.OpenRead(parquetPath);
        using var reader = await ParquetReader.CreateAsync(src, cancellationToken: cancellationToken);

        // Locate the text column by name rather than position so a future
        // schema bump (e.g. adding a row id) doesn't silently grab the
        // wrong column.
        DataField? textField = null;
        foreach (var f in reader.Schema.GetDataFields())
        {
            if (string.Equals(f.Name, "text", StringComparison.Ordinal))
            {
                textField = f;
                break;
            }
        }
        if (textField is null)
        {
            throw new InvalidDataException(
                $"Wikitext parquet schema has no 'text' column. Got: " +
                string.Join(", ", reader.Schema.GetDataFields().Select(f => f.Name)));
        }

        await using var dst = File.Create(outputPath);
        await using var writer = new StreamWriter(dst, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using var rgReader = reader.OpenRowGroupReader(rg);
            var col = await rgReader.ReadColumnAsync(textField, cancellationToken);
            // Parquet.Net surfaces nullable string columns as object?[] /
            // string?[]. Cast to string?[] for direct iteration.
            var data = col.Data as string?[] ?? throw new InvalidDataException(
                $"Wikitext 'text' column had unexpected element type {col.Data.GetType().GetElementType()?.FullName}.");
            foreach (var s in data)
            {
                if (s is null) continue;
                await writer.WriteAsync(s);
            }
        }
    }
}
