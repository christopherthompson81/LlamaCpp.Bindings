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

    // Larger sibling of the default model, used as the "target" in speculative-
    // decoding pair tests. Shares the Qwen3 tokenizer with the default 0.6B,
    // so drafts from the 0.6B model produce ids the 1.7B understands without
    // retokenization. ~1.08 GB — fetched lazily and cached across runs.
    private const string SpecMainEnvVar      = "LLAMACPP_TEST_SPEC_MAIN_MODEL";
    private const string SpecMainDefaultName = "Qwen3-1.7B-UD-Q4_K_XL.gguf";
    private const string SpecMainDownloadUrl = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-UD-Q4_K_XL.gguf";

    // Dedicated embedding model for /v1/embeddings tests. nomic-embed is a
    // widely-deployed 137M-param BERT-style embedding model with a pooling
    // head baked into the GGUF, so it exercises the server's embeddings
    // path end-to-end without relying on a chat model's (nonsense) hidden
    // states. Q4_K_M at ~80 MB keeps the CI download modest.
    private const string EmbedEnvVar      = "LLAMACPP_TEST_EMBEDDING_MODEL";
    private const string EmbedDefaultName = "nomic-embed-text-v1.5.Q4_K_M.gguf";
    private const string EmbedDownloadUrl = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf";

    // Reranker model for /v1/rerank tests. bge-reranker-v2-m3 is the
    // de-facto multilingual reranker; gpustack ships GGUF builds. Q4_K_M
    // is ~418 MB. Reranker model architecture is XLMRoberta-with-rank-head,
    // not exchangeable with the embedding or chat models above.
    private const string RerankEnvVar      = "LLAMACPP_TEST_RERANK_MODEL";
    private const string RerankDefaultName = "bge-reranker-v2-m3-Q4_K_M.gguf";
    private const string RerankDownloadUrl = "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf";

    private static readonly Lazy<string> _path =
        new(Resolve, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<string?> _loraPath =
        new(ResolveLora, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<string?> _specMainPath =
        new(ResolveSpecMain, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<string?> _embedPath =
        new(ResolveEmbed, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<string?> _rerankPath =
        new(ResolveRerank, LazyThreadSafetyMode.ExecutionAndPublication);

    public static string EnsureModelPath() => _path.Value;

    /// <summary>
    /// Returns a LoRA adapter GGUF compatible with the default test base
    /// model, or null if resolution failed (e.g. offline, or
    /// <c>LLAMACPP_TEST_MODEL</c> was overridden to a different architecture
    /// without also setting <c>LLAMACPP_TEST_LORA</c>). Tests should skip
    /// gracefully on null rather than failing.
    /// </summary>
    public static string? TryGetLoraAdapterPath() => _loraPath.Value;

    /// <summary>
    /// Returns a path to a larger Qwen3 model suitable for use as the "main"
    /// side of a speculative-decoding pair (with the default 0.6B test model
    /// as draft). Auto-fetched on first use, reused across runs. Returns
    /// null when the download fails (offline) or when
    /// <c>LLAMACPP_TEST_MODEL</c> has been overridden to a non-Qwen3 model
    /// — in that case the caller must supply
    /// <c>LLAMACPP_TEST_SPEC_MAIN_MODEL</c> explicitly.
    /// </summary>
    public static string? TryGetSpeculativeMainModelPath() => _specMainPath.Value;

    /// <summary>
    /// Returns a path to an embedding GGUF (nomic-embed-text-v1.5 by
    /// default). Auto-fetched on first use, ~80 MB. Returns null if the
    /// download fails — tests should skip rather than fail in that case.
    /// </summary>
    public static string? TryGetEmbeddingModelPath() => _embedPath.Value;

    /// <summary>
    /// Returns a path to a reranker GGUF (bge-reranker-v2-m3 by default).
    /// Auto-fetched on first use, ~418 MB. Returns null on download
    /// failure — tests should skip rather than fail.
    /// </summary>
    public static string? TryGetRerankModelPath() => _rerankPath.Value;

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

    private static string? ResolveSpecMain()
    {
        var env = Environment.GetEnvironmentVariable(SpecMainEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (!File.Exists(env))
                throw new FileNotFoundException(
                    $"{SpecMainEnvVar}='{env}' but the file does not exist.", env);
            return env;
        }

        // Only auto-fetch when the draft side is also the default Qwen3-0.6B
        // — otherwise the vocab won't be compatible and the spec-decoding
        // constructor will reject the pair anyway.
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(EnvVar)))
        {
            return null;
        }

        var cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-test-models");
        Directory.CreateDirectory(cacheDir);

        var dest = Path.Combine(cacheDir, SpecMainDefaultName);
        if (!File.Exists(dest))
        {
            try
            {
                Download(SpecMainDownloadUrl, dest);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TestModelProvider] Speculative main-model download failed: {ex.Message}");
                return null;
            }
        }
        return dest;
    }

    private static string? ResolveEmbed()
    {
        var env = Environment.GetEnvironmentVariable(EmbedEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (!File.Exists(env))
                throw new FileNotFoundException(
                    $"{EmbedEnvVar}='{env}' but the file does not exist.", env);
            return env;
        }

        var cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-test-models");
        Directory.CreateDirectory(cacheDir);

        var dest = Path.Combine(cacheDir, EmbedDefaultName);
        if (!File.Exists(dest))
        {
            try
            {
                Download(EmbedDownloadUrl, dest);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TestModelProvider] Embedding-model download failed: {ex.Message}");
                return null;
            }
        }
        return dest;
    }

    private static string? ResolveRerank()
    {
        var env = Environment.GetEnvironmentVariable(RerankEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (!File.Exists(env))
                throw new FileNotFoundException(
                    $"{RerankEnvVar}='{env}' but the file does not exist.", env);
            return env;
        }

        var cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-test-models");
        Directory.CreateDirectory(cacheDir);

        var dest = Path.Combine(cacheDir, RerankDefaultName);
        if (!File.Exists(dest))
        {
            try
            {
                Download(RerankDownloadUrl, dest);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TestModelProvider] Rerank-model download failed: {ex.Message}");
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
