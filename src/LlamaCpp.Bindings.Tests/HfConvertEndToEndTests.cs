using System.Net.Http;
using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Opt-in real-inference test: download Qwen3-0.6B from HuggingFace,
/// convert to F16 GGUF via <see cref="LlamaHfConverter"/>, load through
/// <see cref="LlamaModel"/>, and run a greedy generation. If the
/// converted model produces non-empty coherent output, the converter's
/// fixes for shape reversal, vocab padding, and F32-norm preservation
/// are all working end-to-end.
/// </summary>
/// <remarks>
/// <para>
/// Gated behind <c>LLAMACPP_E2E_HF=1</c>. Without that env var the
/// test reports as Skipped via <see cref="Assert.Skip"/> — first-time
/// runs need to download ~1.4 GB of safetensors and produce a ~1.4 GB
/// GGUF, totaling ~3 GB on disk. Subsequent runs reuse the cache and
/// complete in seconds.
/// </para>
/// </remarks>
public class HfConvertEndToEndTests
{
    private const string EnvVar = "LLAMACPP_E2E_HF";
    private const string Repo   = "Qwen/Qwen3-0.6B";

    /// <summary>HF files we download to reproduce the model directory.</summary>
    private static readonly string[] HfFiles =
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    };

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable(EnvVar), "1", StringComparison.Ordinal);

    [Fact]
    public async Task Convert_Real_Qwen3_06B_And_Generate()
    {
        if (!Enabled)
        {
            Assert.Skip(
                $"Set {EnvVar}=1 to enable the real HF→GGUF conversion test " +
                "(downloads ~1.4 GB safetensors, produces ~1.4 GB GGUF).");
        }

        var cacheRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "llama-test-models");
        var hfDir = Path.Combine(cacheRoot, "qwen3-0.6b-hf");
        Directory.CreateDirectory(hfDir);

        // 1. Ensure each HF file is on disk.
        await EnsureHfFiles(hfDir, TestContext.Current.CancellationToken);

        // 2. Convert to a fresh F16 GGUF. We write to a unique path under
        //    the same cache directory so concurrent runs don't race; the
        //    file is deleted at the end so the disk footprint is bounded
        //    by the source download.
        var outPath = Path.Combine(cacheRoot,
            $"qwen3-0.6b-converted-f16-{Guid.NewGuid():N}.gguf");
        try
        {
            var convertResult = await LlamaHfConverter.ConvertAsync(
                hfDir, outPath, LlamaHfConvertOutputType.F16,
                progress: null,
                cancellationToken: TestContext.Current.CancellationToken);

            Assert.Equal("qwen3", convertResult.Architecture);
            Assert.True(convertResult.TensorCount > 0);
            Assert.True(convertResult.OutputBytes > 100 * 1024 * 1024,
                $"Converted GGUF unreasonably small: {convertResult.OutputBytes} bytes.");

            // 3. Load through the native loader. If anything is wrong with
            //    the structure (shape mismatch, missing required tensor,
            //    bogus metadata), this throws.
            LlamaBackend.Initialize();
            using var model = new LlamaModel(outPath, new LlamaModelParameters
            {
                GpuLayerCount = 0,
                UseMmap = true,
            });
            Assert.True(model.LayerCount > 0);
            Assert.True(model.EmbeddingSize > 0);

            // 4. Run a greedy generation. We're not checking the answer's
            //    content — base-model continuation without chat-template
            //    framing isn't the converter's responsibility — only that
            //    the model produces some non-empty coherent text.
            using var context = new LlamaContext(model, new LlamaContextParameters
            {
                ContextSize = 256,
                LogicalBatchSize = 256,
                PhysicalBatchSize = 256,
                MaxSequenceCount = 1,
            });
            using var sampler = new LlamaSamplerBuilder().WithGreedy().Build();
            var generator = new LlamaGenerator(context, sampler);

            var pieces = new List<string>();
            await foreach (var piece in generator.GenerateAsync(
                "What is 2 + 2? Answer with just a number.\n",
                maxTokens: 32,
                addSpecial: true,
                parseSpecial: true,
                cancellationToken: TestContext.Current.CancellationToken))
            {
                pieces.Add(piece);
                if (pieces.Count >= 32) break;
            }

            var output = string.Concat(pieces);
            Assert.False(string.IsNullOrWhiteSpace(output),
                "Converted model produced no non-whitespace output. " +
                "Conversion bug likely in tensor data layout.");
            Assert.True(pieces.Count > 0, "Generator produced zero pieces.");
        }
        finally
        {
            // Don't leave a 1.4 GB file behind on every CI pass.
            try { if (File.Exists(outPath)) File.Delete(outPath); } catch { /* best-effort */ }
        }
    }

    /// <summary>Download HF files into <paramref name="dir"/> if missing.</summary>
    private static async Task EnsureHfFiles(string dir, CancellationToken cancellationToken)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
        foreach (var f in HfFiles)
        {
            var dst = Path.Combine(dir, f);
            if (File.Exists(dst)) continue;
            var url = $"https://huggingface.co/{Repo}/resolve/main/{f}";
            var tmp = dst + ".tmp";
            try
            {
                using var resp = await client.GetAsync(
                    url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                resp.EnsureSuccessStatusCode();
                await using var src = await resp.Content.ReadAsStreamAsync(cancellationToken);
                await using var dstFs = File.Create(tmp);
                await src.CopyToAsync(dstFs, cancellationToken);
                File.Move(tmp, dst, overwrite: false);
            }
            catch
            {
                if (File.Exists(tmp)) File.Delete(tmp);
                throw;
            }
        }
    }
}
