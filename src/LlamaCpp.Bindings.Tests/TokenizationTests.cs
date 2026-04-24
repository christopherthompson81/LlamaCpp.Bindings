using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Phase 2 tests: tokenizer round-trip, special token accessors, EOG detection.
/// </summary>
public class TokenizationTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public TokenizationTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Chat_Message_Struct_Size_Matches_Pinned()
    {
        Assert.Equal(16, System.Runtime.CompilerServices.Unsafe.SizeOf<llama_chat_message>());
        Assert.Equal(0, Marshal.OffsetOf<llama_chat_message>(nameof(llama_chat_message.role)).ToInt32());
        Assert.Equal(8, Marshal.OffsetOf<llama_chat_message>(nameof(llama_chat_message.content)).ToInt32());
    }

    [Fact]
    public void Vocab_Reports_Token_Count_And_Special_Tokens()
    {
        var v = _fx.Model.Vocab;

        Assert.True(v.TokenCount > 0, "vocab should have at least one token");
        // A general-purpose chat model must have a BOS and some form of EOG.
        Assert.NotNull(v.Bos);
        Assert.True(v.Eos is not null || v.Eot is not null, "model should have EOS or EOT");
    }

    [Fact]
    public void Tokenize_Roundtrip_Preserves_Content()
    {
        var v = _fx.Model.Vocab;

        // Tokenizer families differ in the edge: Qwen BPE round-trips ASCII
        // verbatim, LLaMA SentencePiece adds a leading space on detokenize
        // (encoded implicitly into the first token). Token-sequence stability
        // under a second round trip is NOT a universal property — on LLaMA,
        // tokenize("X") ≠ tokenize(" X"). The universal property is that
        // the detokenized string equals the input or " " + input.
        const string text = "Hello, world! How are you?";
        var tokens = v.Tokenize(text, addSpecial: false, parseSpecial: false);
        Assert.NotEmpty(tokens);

        var roundtripped = v.Detokenize(tokens);
        Assert.True(roundtripped == text || roundtripped == " " + text,
            $"expected '{text}' or ' {text}'; got '{roundtripped}'");
    }

    [Fact]
    public void Tokenize_AddSpecial_Flag_Is_Honoured_Or_Noop()
    {
        var v = _fx.Model.Vocab;

        // Whether addSpecial actually prepends BOS depends on the model's
        // tokenizer_config (add_bos_token flag). Qwen2/3 have BOS in the vocab
        // but tokenizer_config says false, so the two calls are identical.
        // This test asserts the weaker invariant: addSpecial=true never produces
        // fewer tokens than addSpecial=false, and any extra tokens must be
        // actual special tokens (not garbage).
        var plain    = v.Tokenize("hi", addSpecial: false, parseSpecial: false);
        var withSpec = v.Tokenize("hi", addSpecial: true,  parseSpecial: false);

        Assert.True(withSpec.Length >= plain.Length);
        if (withSpec.Length > plain.Length && v.Bos is int bos)
        {
            // When the model does add special tokens, BOS is the usual suspect.
            Assert.Contains(bos, withSpec);
        }
    }

    [Fact]
    public void IsEndOfGeneration_Returns_True_For_Eos_Or_Eot()
    {
        var v = _fx.Model.Vocab;

        var terminator = v.Eos ?? v.Eot ?? throw new InvalidOperationException("model has no EOS or EOT");
        Assert.True(v.IsEndOfGeneration(terminator));

        // A plainly non-terminator token — the first content token of "hello".
        var helloTokens = v.Tokenize("hello", addSpecial: false, parseSpecial: false);
        Assert.False(v.IsEndOfGeneration(helloTokens[0]));
    }

    [Fact]
    public void TokenToPiece_Returns_NonEmpty_For_Content_Tokens()
    {
        var v = _fx.Model.Vocab;

        var tokens = v.Tokenize("The quick brown fox.", addSpecial: false, parseSpecial: false);
        foreach (var t in tokens)
        {
            var piece = v.TokenToPiece(t, renderSpecial: false);
            Assert.False(string.IsNullOrEmpty(piece), $"token {t} produced empty piece");
        }
    }

    [Fact]
    public void Tokenize_Handles_Empty_Input()
    {
        var v = _fx.Model.Vocab;

        // With addSpecial=false, empty input should produce zero tokens.
        var empty = v.Tokenize(string.Empty, addSpecial: false, parseSpecial: false);
        Assert.Empty(empty);
    }

    [Fact]
    public void Tokenize_Handles_Long_Input_Needing_Retry()
    {
        var v = _fx.Model.Vocab;

        // The point of this test is that the managed Tokenize() correctly
        // handles the "buffer too small, retry with reported size" path:
        // a many-token output well above the 16-token initial buffer, plus
        // a multi-byte UTF-8 character.
        var text = string.Concat(Enumerable.Repeat("naïve rendering ", 500));
        var tokens = v.Tokenize(text, addSpecial: false, parseSpecial: false);
        Assert.True(tokens.Length > 100);

        // Content roundtrip; LLaMA-family tokenizers may add a single leading
        // space, see Tokenize_Roundtrip_Preserves_Content.
        var roundtripped = v.Detokenize(tokens);
        Assert.True(roundtripped == text || roundtripped == " " + text,
            $"long-input roundtrip mismatch (showing first 80 chars): expected '{text[..80]}...' or ' {text[..80]}...'; got '{roundtripped[..Math.Min(80, roundtripped.Length)]}...'");
    }
}

/// <summary>
/// Shared model fixture. Loads the test GGUF once per class. If no model is
/// available it is downloaded automatically via <see cref="TestModelProvider"/>.
/// Set <c>LLAMACPP_TEST_MODEL</c> to override the path.
/// </summary>
public sealed class ModelFixture : IDisposable
{
    public LlamaModel Model { get; }
    public ModelCapabilities Capabilities { get; }

    public ModelFixture()
    {
        var path = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();
        Model = new LlamaModel(path, new LlamaModelParameters
        {
            // Tokenizer + chat-template don't need GPU offload.
            GpuLayerCount = 0,
            UseMmap = true,
        });
        Capabilities = ModelCapabilities.Probe(Model);
    }

    public void Dispose() => Model.Dispose();
}
