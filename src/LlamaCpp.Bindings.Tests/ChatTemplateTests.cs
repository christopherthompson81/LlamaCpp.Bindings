namespace LlamaCpp.Bindings.Tests;

public class ChatTemplateTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fx;
    public ChatTemplateTests(ModelFixture fx) => _fx = fx;

    [Fact]
    public void Model_Exposes_Embedded_Chat_Template()
    {
        if (_fx.Capabilities.SkipUnlessLoaded()) return;
        // Universal: any modern chat model ships a chat template in GGUF metadata.
        // (Pure base / completion / embedding models can lack one — that's why
        // we only assert this once the fixture says one exists.)
        if (!_fx.Capabilities.HasChatTemplate)
        {
            Console.WriteLine($"SKIP: {_fx.Capabilities.DisplayLabel} ships no chat template.");
            return;
        }
        var tmpl = _fx.Model.GetChatTemplate();
        Assert.False(string.IsNullOrWhiteSpace(tmpl));
    }

    [Fact]
    public void Qwen_Chat_Template_Uses_ChatML_Markers()
    {
        if (_fx.Capabilities.SkipUnlessFamily("qwen2", "qwen3")) return;
        // Qwen uses <|im_start|>/<|im_end|> ChatML-style tokens. Pinned so a
        // future llama.cpp change to chat-template extraction would surface here.
        var tmpl = _fx.Model.GetChatTemplate();
        Assert.False(string.IsNullOrWhiteSpace(tmpl));
        Assert.True(tmpl!.Contains("im_start", StringComparison.Ordinal) ||
                    tmpl.Contains("im_end", StringComparison.Ordinal),
            $"Unexpected Qwen chat template shape: {tmpl[..Math.Min(200, tmpl.Length)]}");
    }

    [Fact]
    public void Apply_Template_Produces_Prompt_Containing_Message_Content()
    {
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
