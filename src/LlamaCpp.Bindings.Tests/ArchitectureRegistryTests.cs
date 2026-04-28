namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Smoke tests for <see cref="LlamaArchitectureRegistry"/>: it must
/// pick up every shipped <c>HfConvert/architectures/*.json</c> resource,
/// derive the expected category list, and produce per-layer tensor
/// templates that resolve to canonical GGUF names.
/// </summary>
public class ArchitectureRegistryTests
{
    [Fact]
    public void Registry_LoadsAllEmbeddedArchitectures()
    {
        // At minimum we expect qwen3 (the one we currently ship). New
        // architectures dropped into HfConvert/architectures/ should
        // appear here automatically.
        var all = LlamaArchitectureRegistry.All;
        Assert.NotEmpty(all);
        Assert.Contains("qwen3", all.Keys);
    }

    [Fact]
    public void Qwen3_ProducesExpectedCategories()
    {
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        // The seven canonical categories plus the two top-level tensors.
        // Names match the existing reference profile JSON convention:
        // attn_*.weight (with .weight) for attention tensors,
        // ffn_up/ffn_gate/ffn_down (no .weight) for the FFN block.
        Assert.Contains("attn_q.weight",      spec.Categories);
        Assert.Contains("attn_k.weight",      spec.Categories);
        Assert.Contains("attn_v.weight",      spec.Categories);
        Assert.Contains("attn_output.weight", spec.Categories);
        Assert.Contains("ffn_up",             spec.Categories);
        Assert.Contains("ffn_gate",           spec.Categories);
        Assert.Contains("ffn_down",           spec.Categories);
        Assert.Contains("token_embd.weight",  spec.Categories);
        Assert.Contains("output.weight",      spec.Categories);

        // No norm tensors leak in — they're filtered by IsQuantizableWeight.
        Assert.DoesNotContain(spec.Categories, c => c.Contains("_norm"));
    }

    [Fact]
    public void Qwen3_ExpandPerLayerTensors_ProducesLayerCountTimesTemplates()
    {
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        var resolved = spec.ExpandPerLayerTensors(layerCount: 4).ToList();
        Assert.Equal(4 * spec.PerLayerTensorTemplates.Count, resolved.Count);

        // Every resolved name should start with blk.0./blk.1./blk.2./blk.3.
        // and end in .weight.
        Assert.Contains("blk.0.attn_q.weight", resolved);
        Assert.Contains("blk.3.ffn_down.weight", resolved);
        Assert.All(resolved, n => Assert.EndsWith(".weight", n));
    }

    [Fact]
    public void Lookup_UnknownArch_ReturnsNull()
    {
        Assert.Null(LlamaArchitectureRegistry.Lookup("not-a-real-arch"));
    }

    [Fact]
    public void StandardTransformer_FallbackHasReasonableDefaults()
    {
        var s = LlamaArchitectureRegistry.StandardTransformer;
        Assert.Equal("unknown", s.GgufArch);
        Assert.NotEmpty(s.Categories);
        Assert.NotEmpty(s.PerLayerTensorTemplates);
        // The fallback should cover the seven canonical decoder-only
        // categories — anything less is a bug that would silently
        // under-profile unknown architectures.
        Assert.Contains("ffn_up", s.Categories);
        Assert.Contains("attn_v.weight", s.Categories);
    }
}
