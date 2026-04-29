using System.Runtime.InteropServices;
using LlamaCpp.Bindings.HfConvert;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Pure-logic tests for the Llama-family conversion pieces:
/// the Q/K projection permute (must mirror upstream
/// <c>LlamaModel.permute</c> exactly so K layout under GQA matches
/// llama.cpp's RoPE) and the Llama-3 rope_freqs generator (must mirror
/// the upstream wavelen-band formula so RoPE scaling at long contexts
/// behaves identically).
/// </summary>
public class HfTransformsAndExtraTensorsTests
{
    [Fact]
    public void SelectDefinition_Matches_Llama()
    {
        var def = LlamaHfConverter.SelectDefinition(new[] { "LlamaForCausalLM" });
        Assert.NotNull(def);
        Assert.Equal("llama", def!.GgufArchitecture);
        Assert.Contains(def.TensorMap, t => t.Gguf == "blk.{i}.attn_q.weight" && t.Transform == "permute_q");
        Assert.Contains(def.TensorMap, t => t.Gguf == "blk.{i}.attn_k.weight" && t.Transform == "permute_k");
        Assert.Contains(def.ExtraTensors, e => e.Gguf == "rope_freqs.weight" && e.Generator == "llama3_rope_freqs");
    }

    [Fact]
    public void PermuteQK_Q_Mode_MHA_Swaps_Half_And_Inner_Axes()
    {
        // Tiny MHA case — n_head = 2, head_dim = 4, in_features = 1.
        // Out features = 2 * 4 = 8. Source layout is [n_head=2, 2_halves, half_size=2, 1_in].
        // After swap → [n_head=2, half_size=2, 2_halves, 1_in].
        // Row indices in source vs output:
        //   src layout (h, half, i):  (0,0,0)=0  (0,0,1)=1  (0,1,0)=2  (0,1,1)=3
        //                             (1,0,0)=4  (1,0,1)=5  (1,1,0)=6  (1,1,1)=7
        //   dst layout (h, i, half):  (0,0,0)=0  (0,0,1)=2  (0,1,0)=1  (0,1,1)=3
        //                             (1,0,0)=4  (1,0,1)=6  (1,1,0)=5  (1,1,1)=7
        // Expected permutation: [0, 2, 1, 3, 4, 6, 5, 7].
        var source = new float[] { 0, 1, 2, 3, 4, 5, 6, 7 };
        var bytes = MemoryMarshal.AsBytes(source.AsSpan()).ToArray();
        var (outBytes, outType) = LlamaHfTensorTransforms.PermuteQK(
            bytes, LlamaSafetensorsDtype.F32, LlamaHfConvertOutputType.F32,
            shape: new long[] { 8, 1 },
            nHead: 2, nHeadKv: 2, isK: false);
        var outFloats = MemoryMarshal.Cast<byte, float>(outBytes).ToArray();
        Assert.Equal(0u, outType);  // F32 type id
        Assert.Equal(new float[] { 0, 2, 1, 3, 4, 6, 5, 7 }, outFloats);
    }

    [Fact]
    public void PermuteQK_K_Mode_GQA_Uses_Replace_Not_Divide()
    {
        // Llama-3.2-1B GQA case in miniature: n_head=4, n_kv_head=2.
        // K projection's outer dim factors as n_kv_head × 2 × half_head_dim,
        // so n_groups MUST become 2 (= n_kv_head, REPLACE) not 2 (=4/2,
        // happens to be the same here — pick non-trivial sizes to disambiguate).
        //
        // Use n_head=8, n_kv_head=4. Replace → groups=4, half=8/4/2=1.
        // Divide → groups=8/4=2, half=8/2/2=2. Different layouts; we
        // verify the replace variant.
        //
        // Source [out=8, in=1] = [0..7]. With groups=4, halves=2, half_size=1:
        //   src (h, half, i): each cell is one row.
        //     (0,0,0)=0  (0,1,0)=1  (1,0,0)=2  (1,1,0)=3
        //     (2,0,0)=4  (2,1,0)=5  (3,0,0)=6  (3,1,0)=7
        //   dst (h, i, half) — i has only 1 value, so swapping 2↔1 is a no-op
        //                      when half_size==1. Identity permutation.
        // To exercise a non-identity swap, use half_size=2 → out=8, groups=2.
        // Pick n_head=8, n_kv_head=2 → replace gives groups=2, half=2.
        // Divide would give groups=8/2=4, half=1 (no-op). Distinct outcomes.
        var source = new float[] { 0, 1, 2, 3, 4, 5, 6, 7 };
        var bytes = MemoryMarshal.AsBytes(source.AsSpan()).ToArray();
        var (outBytes, _) = LlamaHfTensorTransforms.PermuteQK(
            bytes, LlamaSafetensorsDtype.F32, LlamaHfConvertOutputType.F32,
            shape: new long[] { 8, 1 },
            nHead: 8, nHeadKv: 2, isK: true);
        var outFloats = MemoryMarshal.Cast<byte, float>(outBytes).ToArray();
        // Replace: groups=2, half=2.
        //   src (h, half, i): (0,0,0)=0 (0,0,1)=1 (0,1,0)=2 (0,1,1)=3
        //                     (1,0,0)=4 (1,0,1)=5 (1,1,0)=6 (1,1,1)=7
        //   dst (h, i, half): (0,0,0)=0 (0,0,1)=2 (0,1,0)=1 (0,1,1)=3
        //                     (1,0,0)=4 (1,0,1)=6 (1,1,0)=5 (1,1,1)=7
        // Expected: [0, 2, 1, 3, 4, 6, 5, 7].
        Assert.Equal(new float[] { 0, 2, 1, 3, 4, 6, 5, 7 }, outFloats);
    }

    [Fact]
    public void PermuteQK_Refuses_Indivisible_Shape()
    {
        var source = new float[] { 0, 1, 2, 3, 4, 5, 6 };  // 7 floats — not divisible by 2*2
        var bytes = MemoryMarshal.AsBytes(source.AsSpan()).ToArray();
        Assert.Throws<InvalidOperationException>(() =>
            LlamaHfTensorTransforms.PermuteQK(
                bytes, LlamaSafetensorsDtype.F32, LlamaHfConvertOutputType.F32,
                shape: new long[] { 7, 1 },
                nHead: 2, nHeadKv: 2, isK: false));
    }

    [Fact]
    public void Llama3RopeFreqs_NoScaling_Returns_Null()
    {
        // Plain Llama-2 / older Llama-3 without llama3-style scaling →
        // generator opts out. Use a config that has rope_theta but no
        // rope_scaling block.
        var json = """
        {
          "architectures": ["LlamaForCausalLM"],
          "hidden_size": 64,
          "num_attention_heads": 4,
          "head_dim": 16,
          "rope_theta": 500000.0
        }
        """;
        using var tmpDir = new TempDir();
        var path = Path.Combine(tmpDir.Path, "config.json");
        File.WriteAllText(path, json);
        using var config = LlamaHfConfig.FromDirectory(tmpDir.Path);

        var result = LlamaHfExtraTensors.Llama3RopeFreqs(config);
        Assert.Null(result);
    }

    [Fact]
    public void Llama3RopeFreqs_Llama32_Config_Produces_Correct_Shape_And_HighFreq_Identity()
    {
        // Real Llama-3.2-1B-Instruct config values.
        var json = """
        {
          "architectures": ["LlamaForCausalLM"],
          "hidden_size": 2048,
          "num_attention_heads": 32,
          "head_dim": 64,
          "rope_theta": 500000.0,
          "rope_scaling": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
          }
        }
        """;
        using var tmpDir = new TempDir();
        File.WriteAllText(Path.Combine(tmpDir.Path, "config.json"), json);
        using var config = LlamaHfConfig.FromDirectory(tmpDir.Path);

        var result = LlamaHfExtraTensors.Llama3RopeFreqs(config);
        Assert.NotNull(result);
        var g = result!.Value;

        // Shape: [head_dim/2] = [32]. F32. 32 floats = 128 bytes.
        Assert.Equal(new long[] { 32 }, g.Shape);
        Assert.Equal(0u, g.TypeId);
        Assert.Equal(128, g.Data.Length);

        var factors = MemoryMarshal.Cast<byte, float>(g.Data).ToArray();
        Assert.Equal(32, factors.Length);

        // Sanity bounds: every factor lives in [1, scale_factor] = [1, 32].
        Assert.All(factors, f => Assert.InRange(f, 1.0f, 32.0f));

        // High-freq tail: the first few indices should be exactly 1.0
        // (wavelen < high_freq_wavelen branch). For Llama-3.2 with
        // base=500000, dim=64: index 0's wavelen = 2π/(1/500000^0) = 2π,
        // far below high_freq_wavelen=2048. → 1.0.
        Assert.Equal(1.0f, factors[0]);
        Assert.Equal(1.0f, factors[1]);

        // Low-freq tail: the last few indices should saturate at scale_factor=32.
        // (wavelen > low_freq_wavelen=8192 branch).
        Assert.Equal(32.0f, factors[^1]);
        Assert.Equal(32.0f, factors[^2]);
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; }
        public TempDir()
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(), $"llama-test-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Path);
        }
        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { /* best-effort */ }
        }
    }
}
