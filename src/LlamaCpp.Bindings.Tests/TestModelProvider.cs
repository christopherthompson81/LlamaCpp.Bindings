namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Resolves the path to the standard test GGUF model, downloading it on first
/// use if it is not already present.
/// <list type="bullet">
///   <item>If <c>LLAMACPP_TEST_MODEL</c> is set, that path is used — a missing
///   file is a hard failure (the caller made an explicit choice).</item>
///   <item>Otherwise the model is fetched once to
///   <c>~/.cache/llama-test-models/</c> and reused across runs.</item>
/// </list>
/// <c>EnsureModelPath()</c> is <see cref="Lazy{T}"/>-guarded so the download
/// happens at most once per test process regardless of how many fixtures call it.
/// </summary>
internal static class TestModelProvider
{
    private const string EnvVar      = "LLAMACPP_TEST_MODEL";
    private const string DefaultName = "Qwen3-0.6B-UD-Q6_K_XL.gguf";
    private const string DownloadUrl = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-Q6_K_XL.gguf";

    // LoRA adapter built against Qwen3-0.6B. 77 MB. Bound to the default
    // base model's architecture — if LLAMACPP_TEST_MODEL is overridden to a
    // non-Qwen3 model, LoRA behavioural tests will need a matching adapter
    // supplied via LLAMACPP_TEST_LORA.
    private const string LoraEnvVar      = "LLAMACPP_TEST_LORA";
    private const string LoraDefaultName = "qwen3-0.6b-lora-test.gguf";
    private const string LoraDownloadUrl = "https://huggingface.co/Chiichen/QWen3-0.6B-Lora-GGUF-Test/resolve/main/Lora-F16-LoRA.gguf";

    private static readonly Lazy<string> _path =
        new(Resolve, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<string?> _loraPath =
        new(ResolveLora, LazyThreadSafetyMode.ExecutionAndPublication);

    public static string EnsureModelPath() => _path.Value;

    /// <summary>
    /// Returns a LoRA adapter GGUF compatible with the default test base
    /// model, or null if resolution failed (e.g. offline, or
    /// <c>LLAMACPP_TEST_MODEL</c> was overridden to a different architecture
    /// without also setting <c>LLAMACPP_TEST_LORA</c>). Tests should skip
    /// gracefully on null rather than failing.
    /// </summary>
    public static string? TryGetLoraAdapterPath() => _loraPath.Value;

    private static string Resolve()
    {
        var env = Environment.GetEnvironmentVariable(EnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (!File.Exists(env))
                throw new FileNotFoundException(
                    $"LLAMACPP_TEST_MODEL='{env}' but the file does not exist.", env);
            return env;
        }

        var cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-test-models");
        Directory.CreateDirectory(cacheDir);

        var dest = Path.Combine(cacheDir, DefaultName);
        if (!File.Exists(dest))
            Download(DownloadUrl, dest);

        return dest;
    }

    private static string? ResolveLora()
    {
        var env = Environment.GetEnvironmentVariable(LoraEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (!File.Exists(env))
                throw new FileNotFoundException(
                    $"{LoraEnvVar}='{env}' but the file does not exist.", env);
            return env;
        }

        // Only auto-fetch the default adapter when the base model is also the
        // default — a custom model may have an incompatible architecture.
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(EnvVar)))
        {
            return null;
        }

        var cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-test-models");
        Directory.CreateDirectory(cacheDir);

        var dest = Path.Combine(cacheDir, LoraDefaultName);
        if (!File.Exists(dest))
        {
            try
            {
                Download(LoraDownloadUrl, dest);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TestModelProvider] LoRA download failed: {ex.Message}");
                return null;
            }
        }
        return dest;
    }

    private static void Download(string url, string dest)
    {
        Console.WriteLine($"[TestModelProvider] Test model not found — downloading to {dest}");
        Console.WriteLine($"[TestModelProvider] Source: {url}");

        var tmp = dest + ".tmp";
        try
        {
            using var client = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromMinutes(30) };
            using var response = client.GetAsync(url, System.Net.Http.HttpCompletionOption.ResponseHeadersRead)
                .GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();

            var total = response.Content.Headers.ContentLength;
            using var src = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
            using var dst = File.Create(tmp);

            var buf = new byte[131_072];
            long written = 0;
            int n;
            while ((n = src.Read(buf, 0, buf.Length)) > 0)
            {
                dst.Write(buf, 0, n);
                written += n;
                if (total.HasValue)
                {
                    int pct = (int)(100L * written / total.Value);
                    Console.Write($"\r[TestModelProvider] {written / 1_048_576:N0} / {total.Value / 1_048_576:N0} MB  ({pct}%)   ");
                }
            }
            Console.WriteLine();
        }
        catch
        {
            if (File.Exists(tmp)) File.Delete(tmp);
            throw;
        }

        File.Move(tmp, dest, overwrite: false);
        Console.WriteLine($"[TestModelProvider] Download complete.");
    }
}
