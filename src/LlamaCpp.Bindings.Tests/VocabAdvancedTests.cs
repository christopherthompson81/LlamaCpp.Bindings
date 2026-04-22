namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Covers the Tier-1 advanced vocab surface: vocab type, per-token attributes,
/// scores, text lookup, tokenizer auto-special flags, and FIM tokens.
///
/// Qwen3 is a chat model (not a code-completion model) so the FIM tests only
/// validate that the plumbing runs and returns null where expected. To actually
/// exercise non-null FIM tokens we'd need a code-completion GGUF (StarCoder,
/// DeepSeek-Coder, CodeLlama, etc.).
/// </summary>
public class VocabAdvancedTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public VocabAdvancedTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void VocabType_Is_Known()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        // Qwen3 is byte-level BPE. But this test is really about "did we plumb
        // the enum correctly" — accept any of the known types.
        var v = _fx.Model.Vocab.VocabType;
        Assert.InRange((int)v, 0, 6);
        Assert.NotEqual(LlamaVocabType.None, v);
    }

    [Fact]
    public void AddsBos_Matches_Previously_Observed_Qwen_Behaviour()
    {
        if (_fx.Capabilities.SkipUnlessFamily("qwen2", "qwen3")) return;
        // Qwen's tokenizer_config has add_bos_token=false even though BOS exists
        // in the vocab. We already observed this during Phase 2 testing — this
        // test pins that observation so a future llama.cpp change would surface
        // it via a test failure.
        var v = _fx.Model!.Vocab;
        Assert.NotNull(v.Bos);
        Assert.False(v.AddsBosAutomatically,
            "Qwen's tokenizer_config has add_bos_token=false; if this fires, either the model changed or llama.cpp's reading of the config did.");
    }

    [Fact]
    public void GetTokenText_Returns_NonEmpty_For_Content_Tokens()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var v = _fx.Model.Vocab;

        var tokens = v.Tokenize("hello", addSpecial: false, parseSpecial: false);
        foreach (var t in tokens)
        {
            var text = v.GetTokenText(t);
            Assert.False(string.IsNullOrEmpty(text),
                $"token {t} should have a non-empty text form");
        }
    }

    [Fact]
    public void GetTokenAttributes_Marks_Content_Tokens_As_Normal()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var v = _fx.Model.Vocab;

        var tokens = v.Tokenize("word", addSpecial: false, parseSpecial: false);
        var first = tokens[0];

        var attrs = v.GetTokenAttributes(first);
        // Content tokens typically have Normal set; special tokens have Control.
        Assert.True((attrs & LlamaTokenAttributes.Normal) != 0
                    || (attrs & LlamaTokenAttributes.UserDefined) != 0
                    || attrs != LlamaTokenAttributes.Undefined,
            $"expected some attribute flag set on content token, got {attrs}");
        Assert.False(v.IsControlToken(first));
    }

    [Fact]
    public void GetTokenAttributes_Marks_Eos_As_Control()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var v = _fx.Model.Vocab;
        var terminator = v.Eos ?? v.Eot;
        Assert.NotNull(terminator);

        Assert.True(v.IsControlToken(terminator!.Value));
        var attrs = v.GetTokenAttributes(terminator.Value);
        Assert.True((attrs & LlamaTokenAttributes.Control) != 0,
            $"EOS token should have Control attribute; got {attrs}");
    }

    [Fact]
    public void GetTokenScore_Is_Finite()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var v = _fx.Model.Vocab;
        var tokens = v.Tokenize("test", addSpecial: false, parseSpecial: false);
        foreach (var t in tokens)
        {
            Assert.False(float.IsNaN(v.GetTokenScore(t)));
            Assert.False(float.IsInfinity(v.GetTokenScore(t)));
        }
    }

    [Fact]
    public void Mask_Token_Is_Null_For_NonMLM_Model()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        // Qwen3 isn't a masked-LM model.
        Assert.Null(_fx.Model.Vocab.Mask);
    }

    [Fact]
    public void FIM_Tokens_Plumbing_Works()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var v = _fx.Model.Vocab;

        // Qwen3 is trained for FIM too (surprising — usually only code models
        // expose these), so at least one of these comes back non-null. The
        // test just validates the plumbing: every value must be either null or
        // a valid in-range token id.
        int[] ids = { v.FimPrefix ?? -1, v.FimSuffix ?? -1, v.FimMiddle ?? -1,
                      v.FimPad ?? -1, v.FimRep ?? -1, v.FimSep ?? -1 };
        foreach (var id in ids)
        {
            if (id == -1) continue; // null → skip
            Assert.InRange(id, 0, v.TokenCount - 1);
            Assert.True(v.IsControlToken(id),
                $"FIM token {id} should be a control token");
        }
    }
}
