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
    public void ResolveTensors_NoFilter_MatchesExpandPerLayerPlusTopLevel()
    {
        // No filters → behave like ExpandPerLayerTensors(layerCount) ∪ TopLevelTensors.
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        var resolved = spec.ResolveTensors(totalLayerCount: 4).ToHashSet();
        var perLayer = spec.ExpandPerLayerTensors(4).ToHashSet();
        Assert.Subset(resolved, perLayer);    // all per-layer expansions present
        foreach (var top in spec.TopLevelTensors)
            Assert.Contains(top, resolved);
    }

    [Fact]
    public void ResolveTensors_CategoryFilter_OnlyEmitsMatchingTemplates()
    {
        // Drill into just attn_v across all 4 layers — no FFN, no top-level.
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        var resolved = spec.ResolveTensors(
            totalLayerCount: 4,
            categoryFilter: new[] { "attn_v.weight" }).ToList();
        Assert.Equal(4, resolved.Count);
        Assert.All(resolved, n => Assert.EndsWith(".attn_v.weight", n));
        Assert.DoesNotContain(resolved, n => n.Contains("ffn_") || n == "output.weight");
    }

    [Fact]
    public void ResolveTensors_LayerFilter_OnlyEmitsSelectedLayers()
    {
        // First-and-last drill: layers 0 and 3 across all per-layer categories.
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        var resolved = spec.ResolveTensors(
            totalLayerCount: 4,
            layerFilter: new[] { 0, 3 }).ToList();

        // Per-layer entries should only have indices 0 or 3.
        var perLayerHits = resolved
            .Where(n => n.StartsWith("blk."))
            .ToList();
        Assert.All(perLayerHits, n =>
            Assert.True(n.StartsWith("blk.0.") || n.StartsWith("blk.3.")));
        // Top-level still appears (filter only applies to per-layer templates).
        Assert.Contains(spec.TopLevelTensors[0], resolved);
    }

    [Fact]
    public void ResolveTensors_BothFilters_ProducesIntersection()
    {
        // The targeted drill that motivated this whole design:
        // "ffn_down on layers [0, 5, 8, 11, 27]" (use_more_bits-like set).
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        var layers = new[] { 0, 5, 8, 11, 27 };
        var resolved = spec.ResolveTensors(
            totalLayerCount: 28,
            categoryFilter: new[] { "ffn_down" },
            layerFilter: layers).ToList();
        Assert.Equal(layers.Length, resolved.Count);
        Assert.All(resolved, n => Assert.EndsWith(".ffn_down.weight", n));
    }

    [Fact]
    public void ResolveTensors_OutOfRangeLayers_AreDroppedSilently()
    {
        // User passes [27] but the model has only 4 layers — drop it,
        // don't error. Lets the UI feed user input without pre-validation.
        var spec = LlamaArchitectureRegistry.Lookup("qwen3");
        Assert.NotNull(spec);

        var resolved = spec.ResolveTensors(
            totalLayerCount: 4,
            layerFilter: new[] { 0, 27, 99 }).ToList();
        var perLayerHits = resolved.Where(n => n.StartsWith("blk.")).ToList();
        Assert.All(perLayerHits, n => Assert.StartsWith("blk.0.", n));
    }

    [Fact]
    public void StripAndNormalizeForCategory_MatchesRegistryConvention()
    {
        // The category strings the registry produces and the inverse helper
        // must agree — round-trip a few templates.
        Assert.Equal("attn_q.weight",
            LlamaArchitectureSpec.StripAndNormalizeForCategory("blk.{i}.attn_q.weight"));
        Assert.Equal("ffn_down",
            LlamaArchitectureSpec.StripAndNormalizeForCategory("blk.{i}.ffn_down.weight"));
        // Resolved-index form too (handy when consumers pass concrete names).
        Assert.Equal("attn_v.weight",
            LlamaArchitectureSpec.StripAndNormalizeForCategory("blk.13.attn_v.weight"));
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
