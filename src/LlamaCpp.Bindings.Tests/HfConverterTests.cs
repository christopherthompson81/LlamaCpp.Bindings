using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end tests for <see cref="LlamaHfConverter"/>. We synthesize a
/// minimal Qwen3-shaped HF model in test setup (config.json,
/// tokenizer.json, model.safetensors), run the converter, and verify
/// the produced GGUF is structurally what consumers expect — every
/// required tensor present, metadata populated, tokenizer round-trips.
/// </summary>
/// <remarks>
/// Synthetic models aren't actually runnable LLMs (the weights are
/// zeros), but the converter only cares about structure: tensor name
/// mapping, dtype conversion, metadata population. A "loads through
/// llama.cpp without crashing" check would need a real-sized model;
/// that's a heavier test we can add later if/when CI bandwidth allows.
/// </remarks>
public class HfConverterTests
{
    [Fact]
    public async Task Convert_Synthetic_Qwen3_Produces_Loadable_Gguf_Structure()
    {
        var dir = MakeTempDir();
        try
        {
            BuildSyntheticQwen3Model(dir,
                hiddenSize: 128, numLayers: 2, numHeads: 4, numKvHeads: 2,
                headDim: 32, ffn: 256, vocabSize: 32);
            var outPath = Path.Combine(dir, "out.gguf");

            var result = await LlamaHfConverter.ConvertAsync(dir, outPath, LlamaHfConvertOutputType.F16);

            Assert.True(File.Exists(outPath));
            Assert.Equal("qwen3", result.Architecture);
            Assert.Equal(LlamaHfConvertOutputType.F16, result.OutputType);
            Assert.True(result.TensorCount > 0);

            // Round-trip through our pure-C# reader.
            var f = LlamaGgufFile.Open(outPath);

            var byKey = f.Metadata.ToDictionary(m => m.Key, m => m.Value);
            Assert.Equal("qwen3",                byKey["general.architecture"].AsString());
            Assert.Equal(128u,                   byKey["qwen3.embedding_length"].AsUInt32());
            Assert.Equal(2u,                     byKey["qwen3.block_count"].AsUInt32());
            Assert.Equal(4u,                     byKey["qwen3.attention.head_count"].AsUInt32());
            Assert.Equal(2u,                     byKey["qwen3.attention.head_count_kv"].AsUInt32());
            Assert.Equal(32u,                    byKey["qwen3.attention.key_length"].AsUInt32());
            Assert.Equal(256u,                   byKey["qwen3.feed_forward_length"].AsUInt32());

            Assert.Equal("gpt2",                 byKey["tokenizer.ggml.model"].AsString());
            Assert.Equal("qwen2",                byKey["tokenizer.ggml.pre"].AsString());
            Assert.True(byKey.ContainsKey("tokenizer.ggml.tokens"));
            Assert.True(byKey.ContainsKey("tokenizer.ggml.token_type"));

            // Every required Qwen3 tensor should be present.
            var tensorNames = new HashSet<string>(f.Tensors.Select(t => t.Name), StringComparer.Ordinal);
            string[] required =
            {
                "token_embd.weight",
                "output_norm.weight",
                "output.weight",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_q_norm.weight",
                "blk.0.attn_k_norm.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
                "blk.1.attn_q.weight",
                "blk.1.ffn_down.weight",
            };
            foreach (var n in required)
            {
                Assert.Contains(n, tensorNames);
            }

            // Dimensions on a couple of representative tensors.
            var qWeight = f.Tensors.First(t => t.Name == "blk.0.attn_q.weight");
            // q_proj shape in HF: [num_heads * head_dim, hidden] = [128, 128]
            // ggml stores in fastest-varying-first order, which we emit by
            // reversing the HF shape — same byte layout, swapped metadata.
            // F16 → 128 * 128 * 2 = 32768 bytes regardless of dim order.
            Assert.Equal(32768, qWeight.ByteSize);

            // 1-D tensors (norms) MUST be F32 even when the chosen output
            // is F16 — ggml's compute path mixes them with F32 activations
            // and rejects an F16 norm with "unsupported types" at decode
            // time. This isn't visible in a structural-only test that
            // never runs inference, so we check the type id directly.
            // F32 = ggml type id 0; F16 = 1.
            foreach (var t in f.Tensors)
            {
                if (t.Dimensions.Length == 1)
                {
                    Assert.Equal(0u, t.TypeId);  // F32
                }
                else
                {
                    Assert.Equal(1u, t.TypeId);  // F16
                }
            }
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Convert_Tied_Embeddings_Skips_Output_Weight()
    {
        var dir = MakeTempDir();
        try
        {
            BuildSyntheticQwen3Model(dir,
                hiddenSize: 64, numLayers: 1, numHeads: 2, numKvHeads: 2,
                headDim: 32, ffn: 128, vocabSize: 16,
                tieWordEmbeddings: true,
                writeLmHead: false);
            var outPath = Path.Combine(dir, "out-tied.gguf");

            await LlamaHfConverter.ConvertAsync(dir, outPath, LlamaHfConvertOutputType.F16);

            var f = LlamaGgufFile.Open(outPath);
            var tensorNames = new HashSet<string>(f.Tensors.Select(t => t.Name), StringComparer.Ordinal);
            Assert.Contains("token_embd.weight", tensorNames);
            Assert.DoesNotContain("output.weight", tensorNames);
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public async Task Convert_F32_Output_Is_Lossless_From_F16_Input()
    {
        var dir = MakeTempDir();
        try
        {
            BuildSyntheticQwen3Model(dir,
                hiddenSize: 64, numLayers: 1, numHeads: 2, numKvHeads: 2,
                headDim: 32, ffn: 128, vocabSize: 16,
                dtype: "F16");
            var outPath = Path.Combine(dir, "out-f32.gguf");

            await LlamaHfConverter.ConvertAsync(dir, outPath, LlamaHfConvertOutputType.F32);
            var f = LlamaGgufFile.Open(outPath);
            // Every tensor should be F32 type id (0).
            foreach (var t in f.Tensors)
            {
                Assert.Equal(0u, t.TypeId);
            }
        }
        finally
        {
            DeleteDir(dir);
        }
    }

    [Fact]
    public void SelectDefinition_Matches_Qwen3()
    {
        var def = LlamaHfConverter.SelectDefinition(new[] { "Qwen3ForCausalLM" });
        Assert.NotNull(def);
        Assert.Equal("qwen3", def!.GgufArchitecture);
    }

    [Fact]
    public void SelectDefinition_Returns_Null_For_Unknown_Architecture()
    {
        Assert.Null(LlamaHfConverter.SelectDefinition(new[] { "NonexistentArchitecture42" }));
    }

    // ----- Synthetic model builder -----

    /// <summary>
    /// Build a minimal Qwen3-shaped HF model layout in
    /// <paramref name="dir"/>: <c>config.json</c>, <c>tokenizer.json</c>,
    /// <c>tokenizer_config.json</c>, <c>model.safetensors</c>. All
    /// weight tensors are zero-filled — the test only validates the
    /// converter's structural transformations, not numerical fidelity.
    /// </summary>
    private static void BuildSyntheticQwen3Model(
        string dir,
        int hiddenSize, int numLayers, int numHeads, int numKvHeads,
        int headDim, int ffn, int vocabSize,
        bool tieWordEmbeddings = false, bool writeLmHead = true,
        string dtype = "BF16")
    {
        // config.json
        var config = new Dictionary<string, object?>
        {
            ["architectures"]              = new[] { "Qwen3ForCausalLM" },
            ["hidden_size"]                = hiddenSize,
            ["num_hidden_layers"]          = numLayers,
            ["num_attention_heads"]        = numHeads,
            ["num_key_value_heads"]        = numKvHeads,
            ["head_dim"]                   = headDim,
            ["intermediate_size"]          = ffn,
            ["max_position_embeddings"]    = 256,
            ["rms_norm_eps"]               = 1e-6,
            ["rope_theta"]                 = 10000.0,
            ["tie_word_embeddings"]        = tieWordEmbeddings,
            ["vocab_size"]                 = vocabSize,
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        // tokenizer.json — minimal BPE with vocabSize tokens and no merges.
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < vocabSize; i++)
        {
            // Use unique strings; "tok_0", "tok_1", ... avoids the BPE
            // model rejecting duplicate keys.
            vocab[$"tok_{i}"] = i;
        }
        var tokenizerJson = new Dictionary<string, object?>
        {
            ["version"] = "1.0",
            ["truncation"] = (object?)null,
            ["padding"] = (object?)null,
            ["added_tokens"] = new[]
            {
                new Dictionary<string, object?>
                {
                    ["id"] = vocabSize - 1,
                    ["content"] = "<|endoftext|>",
                    ["single_word"] = false,
                    ["lstrip"] = false,
                    ["rstrip"] = false,
                    ["normalized"] = false,
                    ["special"] = true,
                },
            },
            ["model"] = new Dictionary<string, object?>
            {
                ["type"] = "BPE",
                ["dropout"] = (object?)null,
                ["unk_token"] = (object?)null,
                ["continuing_subword_prefix"] = (object?)null,
                ["end_of_word_suffix"] = (object?)null,
                ["fuse_unk"] = false,
                ["vocab"] = vocab,
                ["merges"] = Array.Empty<string>(),
            },
        };
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"),
            JsonSerializer.Serialize(tokenizerJson, new JsonSerializerOptions { WriteIndented = true }));

        // tokenizer_config.json — bos/eos hooks for the special-id resolver.
        var tokConfig = new Dictionary<string, object?>
        {
            ["bos_token"] = "<|endoftext|>",
            ["eos_token"] = "<|endoftext|>",
        };
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
            JsonSerializer.Serialize(tokConfig));

        // model.safetensors — synthesize the Qwen3 tensor stack with
        // zero data. Layout per-tensor: [shape], dtype string, byte
        // offset relative to the data section.
        var tensors = new List<(string name, int[] shape)>();
        // Embedding + output.
        tensors.Add(("model.embed_tokens.weight", new[] { vocabSize, hiddenSize }));
        tensors.Add(("model.norm.weight",        new[] { hiddenSize }));
        if (writeLmHead)
        {
            tensors.Add(("lm_head.weight", new[] { vocabSize, hiddenSize }));
        }
        // Per-layer.
        for (int i = 0; i < numLayers; i++)
        {
            tensors.Add(($"model.layers.{i}.input_layernorm.weight",          new[] { hiddenSize }));
            tensors.Add(($"model.layers.{i}.self_attn.q_proj.weight",         new[] { numHeads * headDim,    hiddenSize }));
            tensors.Add(($"model.layers.{i}.self_attn.k_proj.weight",         new[] { numKvHeads * headDim,  hiddenSize }));
            tensors.Add(($"model.layers.{i}.self_attn.v_proj.weight",         new[] { numKvHeads * headDim,  hiddenSize }));
            tensors.Add(($"model.layers.{i}.self_attn.q_norm.weight",         new[] { headDim }));
            tensors.Add(($"model.layers.{i}.self_attn.k_norm.weight",         new[] { headDim }));
            tensors.Add(($"model.layers.{i}.self_attn.o_proj.weight",         new[] { hiddenSize, numHeads * headDim }));
            tensors.Add(($"model.layers.{i}.post_attention_layernorm.weight", new[] { hiddenSize }));
            tensors.Add(($"model.layers.{i}.mlp.gate_proj.weight",            new[] { ffn,        hiddenSize }));
            tensors.Add(($"model.layers.{i}.mlp.up_proj.weight",              new[] { ffn,        hiddenSize }));
            tensors.Add(($"model.layers.{i}.mlp.down_proj.weight",            new[] { hiddenSize, ffn }));
        }

        WriteSyntheticSafetensors(Path.Combine(dir, "model.safetensors"), tensors, dtype);
    }

    /// <summary>
    /// Hand-roll a safetensors file with zeroed data. Format:
    /// 8-byte header length + UTF-8 JSON header + tensor data.
    /// </summary>
    private static void WriteSyntheticSafetensors(
        string path, IReadOnlyList<(string name, int[] shape)> tensors, string dtype)
    {
        int bytesPerElement = dtype switch
        {
            "F32"  => 4,
            "F16"  => 2,
            "BF16" => 2,
            _ => throw new ArgumentException($"Unsupported synthetic dtype '{dtype}'.", nameof(dtype)),
        };

        var header = new Dictionary<string, object?>();
        long offset = 0;
        foreach (var (name, shape) in tensors)
        {
            long n = 1;
            foreach (var d in shape) n *= d;
            long size = n * bytesPerElement;
            header[name] = new Dictionary<string, object?>
            {
                ["dtype"]        = dtype,
                ["shape"]        = shape.Select(d => (long)d).ToArray(),
                ["data_offsets"] = new long[] { offset, offset + size },
            };
            offset += size;
        }
        var headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));

        using var fs = File.Create(path);
        Span<byte> lenBytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(lenBytes, (ulong)headerBytes.Length);
        fs.Write(lenBytes);
        fs.Write(headerBytes);

        // Zero data section.
        var zeros = new byte[64 * 1024];
        long remaining = offset;
        while (remaining > 0)
        {
            int chunk = (int)Math.Min(zeros.Length, remaining);
            fs.Write(zeros, 0, chunk);
            remaining -= chunk;
        }
    }

    private static string MakeTempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "llama-hfconvert-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void DeleteDir(string d)
    {
        try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { /* best-effort */ }
    }
}
