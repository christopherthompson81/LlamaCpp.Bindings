using LlamaCpp.Bindings.LlamaChat.Services.Remote;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end tests for the LlamaChat client that talks to
/// <see cref="LlamaCpp.Bindings.Server"/>. Reuses
/// <see cref="LlamaServerTests.Factory"/> so we share the expensive model
/// load. Each test pushes the request through the client (which constructs
/// its own <c>HttpClient</c>) by injecting the factory's in-process test
/// handler — that way these tests verify the client's URL composition,
/// header handling, and SSE drain against the real server pipeline rather
/// than a mock.
/// </summary>
public class OpenAiChatClientTests : IClassFixture<LlamaServerTests.Factory>
{
    private readonly LlamaServerTests.Factory _factory;
    public OpenAiChatClientTests(LlamaServerTests.Factory factory) => _factory = factory;

    /// <summary>
    /// Build an <see cref="OpenAiChatClient"/> whose underlying HttpClient
    /// is wired to the in-process test server. The base URL has no trailing
    /// slash on purpose — that's the configuration that triggered the
    /// double-slash 404 bug we're regression-testing.
    /// </summary>
    private OpenAiChatClient NewClient(string? apiKey = null) =>
        new("http://localhost", apiKey, _factory.Server.CreateHandler());

    [Fact]
    public async Task ListModels_Hits_V1Models_Without_Double_Slash()
    {
        // Regression: a base URL without a trailing slash was producing
        // "//v1/models" because Uri.ToString() re-adds one. The server's
        // router 404s on that, so a passing call here is sufficient to
        // pin the fix.
        using var client = NewClient();
        var ids = await client.ListModelsAsync(TestContext.Current.CancellationToken);
        Assert.NotEmpty(ids);
        Assert.False(string.IsNullOrEmpty(ids[0]));
    }

    [Fact]
    public async Task CreateChatCompletion_NonStreaming_Returns_Assistant_Content()
    {
        using var client = NewClient();
        var resp = await client.CreateChatCompletionAsync(new ChatCompletionsRequest
        {
            Messages = new()
            {
                new() { Role = "user", Content = MessageContent.FromText("Say hi.") },
            },
            MaxTokens = 16,
        }, TestContext.Current.CancellationToken);

        Assert.NotNull(resp);
        Assert.Single(resp.Choices);
        Assert.Equal("assistant", resp.Choices[0].Message.Role);
        Assert.False(string.IsNullOrWhiteSpace(resp.Choices[0].Message.Content?.Text ?? ""));
    }

    [Fact]
    public async Task StreamChatCompletion_Yields_Multiple_Chunks_And_Terminates()
    {
        using var client = NewClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new()
            {
                new() { Role = "user", Content = MessageContent.FromText("Count to five.") },
            },
            MaxTokens = 32,
        };

        int chunkCount = 0;
        var sb = new System.Text.StringBuilder();
        await foreach (var chunk in client.StreamChatCompletionAsync(req, TestContext.Current.CancellationToken))
        {
            chunkCount++;
            if (chunk.Choices.Count > 0 && chunk.Choices[0].Delta.Content is { } delta)
                sb.Append(delta);
        }

        Assert.True(chunkCount > 1, $"expected multiple chunks; got {chunkCount}");
        Assert.False(string.IsNullOrWhiteSpace(sb.ToString()),
            "streamed content was empty");
    }

    [Fact]
    public async Task ListModels_Tolerates_Trailing_Slash_On_BaseUrl()
    {
        // Symmetry check for the URL normalization: a base URL *with* a
        // trailing slash should also produce a clean "/v1/models" path.
        using var client = new OpenAiChatClient(
            "http://localhost/", apiKey: null, _factory.Server.CreateHandler());
        var ids = await client.ListModelsAsync(TestContext.Current.CancellationToken);
        Assert.NotEmpty(ids);
    }
}
