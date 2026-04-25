using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using LlamaCpp.Bindings.Server;
using LlamaCpp.Bindings.Server.Models;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// End-to-end smoke tests for <see cref="LlamaCpp.Bindings.Server"/>. Spins
/// up the server in-process via <see cref="WebApplicationFactory{TEntryPoint}"/>
/// with the default test GGUF, then exercises the four V1 endpoints against
/// the real HTTP pipeline. These are expensive (they load a model) but fast
/// enough on a 3090 to be worth running on every build.
/// </summary>
public class LlamaServerTests : IClassFixture<LlamaServerTests.Factory>
{
    private readonly Factory _factory;
    public LlamaServerTests(Factory factory) => _factory = factory;

    // ----- /health -----

    [Fact]
    public async Task Health_Returns_Ok()
    {
        var client = _factory.CreateClient();
        var resp = await client.GetAsync("/health", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
    }

    // ----- /v1/models -----

    [Fact]
    public async Task V1Models_Reports_Loaded_Model()
    {
        var client = _factory.CreateClient();
        var resp = await client.GetFromJsonAsync<ModelsListResponse>(
            "/v1/models", TestContext.Current.CancellationToken);
        Assert.NotNull(resp);
        Assert.NotEmpty(resp!.Data);
        Assert.Equal("list", resp.Object);
        Assert.False(string.IsNullOrEmpty(resp.Data[0].Id));
    }

    // ----- /v1/chat/completions (non-streaming) -----

    [Fact]
    public async Task ChatCompletions_NonStreaming_Returns_Content()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Model = "ignored-in-v1",
            Messages = new()
            {
                new() { Role = "user", Content = "Say hi." },
            },
            MaxTokens = 16,
            Stream = false,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotNull(body);
        Assert.Single(body!.Choices);
        Assert.Equal("assistant", body.Choices[0].Message.Role);
        Assert.False(string.IsNullOrWhiteSpace(body.Choices[0].Message.Content),
            "assistant message content was empty");
    }

    // ----- /v1/chat/completions (streaming SSE) -----

    [Fact]
    public async Task ChatCompletions_Streaming_Yields_Multiple_Chunks_And_Terminates()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new()
            {
                new() { Role = "user", Content = "Count to five." },
            },
            MaxTokens = 32,
            Stream = true,
        };

        using var httpReq = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
        {
            Content = JsonContent.Create(req),
        };
        using var resp = await client.SendAsync(
            httpReq, HttpCompletionOption.ResponseHeadersRead, TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.StartsWith("text/event-stream", resp.Content.Headers.ContentType?.MediaType ?? "");

        using var stream = await resp.Content.ReadAsStreamAsync(TestContext.Current.CancellationToken);
        using var reader = new StreamReader(stream);

        int dataChunks = 0;
        bool sawDone = false;
        var content = new StringBuilder();
        string? role = null;

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync(TestContext.Current.CancellationToken);
            if (line is null) break;
            if (!line.StartsWith("data: ")) continue;
            var payload = line.Substring("data: ".Length);
            if (payload == "[DONE]") { sawDone = true; break; }

            dataChunks++;
            using var doc = JsonDocument.Parse(payload);
            var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");
            if (delta.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String)
            {
                role = r.GetString();
            }
            if (delta.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
            {
                content.Append(c.GetString());
            }
        }

        Assert.True(sawDone, "stream did not terminate with [DONE]");
        Assert.True(dataChunks >= 2, $"expected multiple chunks, got {dataChunks}");
        Assert.Equal("assistant", role);
        Assert.False(string.IsNullOrWhiteSpace(content.ToString()),
            "streamed content was empty");
    }

    // ----- /completion (native llama-server endpoint) -----

    [Fact]
    public async Task RawCompletion_NonStreaming_Returns_Content()
    {
        var client = _factory.CreateClient();
        var req = new CompletionRequest
        {
            Prompt = "The capital of France is",
            MaxTokens = 6,
            Stream = false,
        };
        var resp = await client.PostAsJsonAsync(
            "/completion", req, TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

        var body = await resp.Content.ReadFromJsonAsync<CompletionResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotNull(body);
        Assert.False(string.IsNullOrWhiteSpace(body!.Content));
    }

    // ----- Concurrency: two in-flight chat completions don't corrupt each other. -----

    [Fact]
    public async Task Two_Concurrent_Chat_Completions_Both_Succeed()
    {
        // With MaxSequenceCount = 2 (configured by the factory) both requests
        // should run in parallel via distinct LlamaSessions. Serializing
        // through the decode lock is expected; crashing, deadlocking, or
        // cross-contaminating each other is not.
        var client = _factory.CreateClient();

        async Task<string> Run(string prompt)
        {
            var req = new ChatCompletionsRequest
            {
                Messages = new() { new() { Role = "user", Content = prompt } },
                MaxTokens = 12,
                Stream = false,
            };
            var resp = await client.PostAsJsonAsync(
                "/v1/chat/completions", req, TestContext.Current.CancellationToken);
            resp.EnsureSuccessStatusCode();
            var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
                cancellationToken: TestContext.Current.CancellationToken);
            return body!.Choices[0].Message.Content;
        }

        var results = await Task.WhenAll(
            Run("Name a colour."),
            Run("Name a fruit."));

        Assert.False(string.IsNullOrWhiteSpace(results[0]));
        Assert.False(string.IsNullOrWhiteSpace(results[1]));
    }

    // ----- fixture -----

    public sealed class Factory : WebApplicationFactory<Program>
    {
        protected override IHost CreateHost(IHostBuilder builder)
        {
            // Resolve the model path + inject server options before the
            // test host boots its Options pipeline. Using AddInMemoryCollection
            // lets the test override what appsettings.json ships with.
            var modelPath = TestModelProvider.EnsureModelPath();
            builder.ConfigureAppConfiguration(cfg =>
            {
                cfg.AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["LlamaServer:ModelPath"]         = modelPath,
                    ["LlamaServer:ContextSize"]       = "2048",
                    ["LlamaServer:MaxSequenceCount"]  = "2",
                    ["LlamaServer:GpuLayerCount"]     = "-1",
                    ["LlamaServer:OffloadKqv"]        = "true",
                    ["LlamaServer:MaxOutputTokens"]   = "64",
                    // Let WebApplicationFactory handle the listening URL; the
                    // test client bypasses real sockets anyway.
                    ["LlamaServer:Urls"]              = "",
                });
            });
            return base.CreateHost(builder);
        }
    }
}
