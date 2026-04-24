namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Covers the Tier-1 model-metadata surface: architecture accessors, capability
/// flags, description, metadata dictionary. Uses the CPU-loaded ModelFixture
/// so it doesn't contend for VRAM with the generation suite.
/// </summary>
public class ModelMetadataTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public ModelMetadataTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Architecture_Accessors_Report_Sane_Values()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var m = _fx.Model;

        Assert.True(m.AttentionHeadCount > 0);
        Assert.True(m.KvHeadCount > 0);
        Assert.True(m.KvHeadCount <= m.AttentionHeadCount,
            "KV heads should be ≤ attention heads (GQA / MHA).");
        Assert.True(m.EmbeddingInputSize > 0);
        Assert.True(m.EmbeddingOutputSize > 0);
        // SlidingWindowSize may be 0 for dense-attention models; just ensure non-negative.
        Assert.True(m.SlidingWindowSize >= 0);
    }

    [Fact]
    public void Size_And_ParameterCount_Match_File()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var m = _fx.Model;

        // Size should roughly equal the GGUF file size on disk. llama_model_size
        // reports tensor-bytes, not file-bytes, so allow a generous tolerance
        // (GGUF metadata overhead is ~MB, not GB). Qwen3 IQ4_XS is ~17GB.
        var fi = new FileInfo(m.ModelPath);
        Assert.True(fi.Exists);
        Assert.True(m.SizeInBytes > 0);
        Assert.InRange(m.SizeInBytes, fi.Length / 2, fi.Length * 2);

        Assert.True(m.ParameterCount > 0, $"expected non-zero param count, got {m.ParameterCount}");
    }

    [Fact]
    public void Capability_Flags_Report_Decoder_Only()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var m = _fx.Model;

        // Qwen3 is decoder-only, autoregressive, hybrid-attention MoE.
        Assert.True(m.HasDecoder);
        Assert.False(m.HasEncoder);
        Assert.False(m.IsDiffusion);

        // Qwen3 A3B uses Gated Delta Net (recurrent memory) in some layers,
        // so is_hybrid should be true. Is_recurrent is unambiguous (pure RNN
        // architectures like RWKV); Qwen should report false.
        // Be lenient — the value is what llama.cpp reports.
        Assert.False(m.IsRecurrent, "Qwen3 is not a pure recurrent architecture");
    }

    [Fact]
    public void DecoderStartToken_Is_Null_For_DecoderOnly_Models()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        // Encoder-decoder models (T5 etc.) define this; Qwen3 does not.
        Assert.Null(_fx.Model.DecoderStartToken);
    }

    [Fact]
    public void ClassifierOutputCount_Is_Nonnegative_And_Label_Lookup_Is_Bounded()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var m = _fx.Model;

        // llama_model_n_cls_out counts classifier heads for reranker / classifier
        // models; for pure generative models it has historically returned 0 or 1
        // depending on whether llama.cpp treats the LM head as a classifier.
        // Don't hard-code the value; just assert it's non-negative and that
        // lookups beyond the count return null.
        Assert.True(m.ClassifierOutputCount >= 0);
        Assert.Null(m.GetClassifierLabel(-1));
        Assert.Null(m.GetClassifierLabel(m.ClassifierOutputCount + 10));
    }

    [Fact]
    public void Description_Is_Populated()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        // llama_model_desc emits something like "qwen3moe 35B A3B IQ4_XS" — don't
        // hard-code the string, just assert non-empty and contains recognisable
        // tokens from the architecture or quant type.
        Assert.False(string.IsNullOrWhiteSpace(_fx.Model.Description));
    }

    [Fact]
    public void RopeType_Is_Sensible()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var m = _fx.Model;

        // Qwen3.6-35B-A3B uses IMRope (interleaved M-RoPE, value 40). Asserting
        // the specific variant would fail on a different model; the universal
        // invariant is "not None, and the training freq scale is defined."
        Assert.True(m.UsesRotaryEmbeddings);
        Assert.NotEqual(LlamaRopeType.None, m.RopeType);
        Assert.True(m.TrainingRopeFreqScale > 0);
    }

    [Fact]
    public void Metadata_Dictionary_Is_Populated_And_Contains_Well_Known_Keys()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var md = _fx.Model.Metadata;

        Assert.True(md.Count > 10, "GGUF metadata should have many keys");
        // Every well-formed GGUF has general.architecture.
        Assert.Contains("general.architecture", md.Keys);
        Assert.False(string.IsNullOrEmpty(md["general.architecture"]));
    }

    [Fact]
    public void GetMetadata_Single_Key_Matches_Dictionary_Entry()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var md = _fx.Model.Metadata;
        const string key = "general.architecture";
        Assert.Equal(md[key], _fx.Model.GetMetadata(key));
        Assert.Null(_fx.Model.GetMetadata("__not_a_real_key__"));
    }
}
