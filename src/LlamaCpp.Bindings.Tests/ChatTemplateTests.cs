namespace LlamaCpp.Bindings.Tests;

public class ChatTemplateTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public ChatTemplateTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Model_Exposes_Embedded_Chat_Template()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var tmpl = _fx.Model.GetChatTemplate();
        Assert.False(string.IsNullOrWhiteSpace(tmpl),
            "Qwen3-class chat model should ship a chat template in GGUF metadata.");
        // Qwen uses <|im_start|>/<|im_end|> ChatML-style tokens — the template
        // should reference at least one of them.
        Assert.True(tmpl!.Contains("im_start", StringComparison.Ordinal) ||
                    tmpl.Contains("im_end", StringComparison.Ordinal),
            $"Unexpected chat template shape: {tmpl[..Math.Min(200, tmpl.Length)]}");
    }

    [Fact]
    public void Apply_Template_Produces_Prompt_Containing_Message_Content()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var tmpl = _fx.Model.GetChatTemplate();
        Assert.NotNull(tmpl);

        var messages = new[]
        {
            new ChatMessage("system", "You are a helpful assistant."),
            new ChatMessage("user",   "What is the capital of France?"),
        };

        var prompt = LlamaChatTemplate.Apply(tmpl!, messages, addAssistantPrefix: true);
        Assert.False(string.IsNullOrWhiteSpace(prompt));
        Assert.Contains("helpful assistant", prompt, StringComparison.Ordinal);
        Assert.Contains("capital of France", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void Apply_Template_With_And_Without_Assistant_Prefix_Differ()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var tmpl = _fx.Model.GetChatTemplate();
        Assert.NotNull(tmpl);

        var messages = new[]
        {
            new ChatMessage("user", "ping"),
        };

        var withPrefix    = LlamaChatTemplate.Apply(tmpl!, messages, addAssistantPrefix: true);
        var withoutPrefix = LlamaChatTemplate.Apply(tmpl!, messages, addAssistantPrefix: false);

        Assert.NotEqual(withPrefix, withoutPrefix);
        // The "with prefix" version should be longer — it ends with the
        // assistant-turn opener (e.g. "<|im_start|>assistant\n").
        Assert.True(withPrefix.Length > withoutPrefix.Length);
    }

    [Fact]
    public void Template_Output_Tokenizes_With_parseSpecial_true()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }
        var tmpl = _fx.Model.GetChatTemplate();
        Assert.NotNull(tmpl);

        var messages = new[]
        {
            new ChatMessage("user", "hello"),
        };
        var prompt = LlamaChatTemplate.Apply(tmpl!, messages, addAssistantPrefix: true);

        // Tokenize the templated prompt as the generator would: parseSpecial=true
        // so <|im_start|> etc. turn into their canonical single-token form rather
        // than being chopped into literal text.
        var tokens = _fx.Model.Vocab.Tokenize(prompt, addSpecial: false, parseSpecial: true);
        Assert.True(tokens.Length > 0);
        Assert.True(tokens.Length < prompt.Length, "parseSpecial should collapse special markers into single tokens");
    }

    [Fact]
    public void Apply_Throws_On_Empty_Messages()
    {
        LlamaBackend.Initialize();
        Assert.Throws<ArgumentException>(() =>
            LlamaChatTemplate.Apply("anything", Array.Empty<ChatMessage>()));
    }
}
