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

    // ----- Prompt caching: X-Cached-Tokens + reuse across requests -----

    [Fact]
    public async Task First_Request_With_Unique_Prompt_Reports_Tiny_Cache_Hit()
    {
        // A unique prompt can't match the user-authored content of any
        // prior slot, but the chat template's leading role-preamble tokens
        // (e.g. `<|im_start|>user\n` on Qwen3, ~3 tokens) are identical
        // across every user turn and therefore always in some slot's KV
        // after the first request to the server. So "cold" really means
        // "hits only the fixed template preamble" — a handful of tokens,
        // not zero. If we see a big number here, either uniqueness is
        // broken or another test has primed a matching body.
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Unique-prompt-{Guid.NewGuid():N}" } },
            MaxTokens = 4,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        int cached = int.Parse(resp.Headers.GetValues("X-Cached-Tokens").First());
        Assert.True(cached < 10,
            $"Unique-content prompt should only match the chat template preamble; got {cached} tokens cached.");
    }

    [Fact]
    public async Task Identical_Followup_Request_Hits_The_Prompt_Cache()
    {
        // Two identical requests back-to-back. The second must reuse almost
        // every prompt token — firstNewIndex backs off one token at the tail
        // so llama_decode has something to do, so expect (promptLen - 1)
        // cached tokens. A zero or near-zero report means the cache is
        // broken; a non-zero report proves the pool hit path works.
        // Guid-suffixed content so earlier tests in the class can't have
        // primed a matching slot — first request must genuinely be a miss.
        var marker = Guid.NewGuid().ToString("N");
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Name a fruit ({marker})." } },
            MaxTokens = 4,
        };

        var first = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        first.EnsureSuccessStatusCode();
        int firstCached = int.Parse(first.Headers.GetValues("X-Cached-Tokens").First());

        var second = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        second.EnsureSuccessStatusCode();
        int secondCached = int.Parse(second.Headers.GetValues("X-Cached-Tokens").First());

        // Second request must have dramatically more cached tokens than the
        // first — identical prompt, prior slot still idle, so the entire
        // prompt (minus the 1-token tail that the generator re-decodes) is
        // a cache hit. A floor of "at least 10 more than the first" easily
        // clears the few-token template preamble that also matches on the
        // cold run.
        Assert.True(secondCached >= firstCached + 10,
            $"Identical second request should reuse much more of the cache; first={firstCached}, second={secondCached}.");
    }

    [Fact]
    public async Task Multi_Turn_Conversation_Reuses_Prefix()
    {
        // Classic chat scenario: turn 1 builds up a conversation; turn 2
        // sends the whole conversation + one new user message. The rendered
        // template for turn 2 starts with exactly the same bytes as turn 1's
        // prompt, so the pool should find a large LCP.
        var client = _factory.CreateClient();
        var marker = Guid.NewGuid().ToString("N");

        var turn1 = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"What is 2 + 2? ({marker})" } },
            MaxTokens = 6,
        };
        var r1 = await client.PostAsJsonAsync(
            "/v1/chat/completions", turn1, TestContext.Current.CancellationToken);
        r1.EnsureSuccessStatusCode();
        var body1 = await r1.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        var assistantReply = body1!.Choices[0].Message.Content;
        int turn1Cached = int.Parse(r1.Headers.GetValues("X-Cached-Tokens").First());
        // Turn 1's unique marker keeps the body miss-only; template
        // preamble (~3 tokens) is the only legitimate cache hit here.
        Assert.True(turn1Cached < 10,
            $"Turn 1 should be cold aside from the template preamble; cached={turn1Cached}.");

        var turn2 = new ChatCompletionsRequest
        {
            Messages = new()
            {
                new() { Role = "user",      Content = $"What is 2 + 2? ({marker})" },
                new() { Role = "assistant", Content = assistantReply },
                new() { Role = "user",      Content = "Now what is 3 + 3?" },
            },
            MaxTokens = 6,
        };
        var r2 = await client.PostAsJsonAsync(
            "/v1/chat/completions", turn2, TestContext.Current.CancellationToken);
        r2.EnsureSuccessStatusCode();
        int turn2Cached = int.Parse(r2.Headers.GetValues("X-Cached-Tokens").First());

        // Exact cached-token count depends on the chat template. For any
        // reasonable Jinja template, turn 2's prompt starts with turn 1's
        // rendered user message verbatim, so cached must be at least as
        // large as the tokenized "What is 2 + 2?" framing — tens of tokens.
        // A floor of 10 is generous; a broken pool produces 0.
        Assert.True(turn2Cached >= 10,
            $"Expected multi-turn chat to reuse a substantial prefix; X-Cached-Tokens={turn2Cached}.");
    }

    [Fact]
    public async Task Different_Prompt_Does_Not_Falsely_Hit_Cache()
    {
        // After priming a slot with prompt A, a subsequent prompt B with
        // no shared prefix must report 0 cached tokens — the matching
        // walk shouldn't invent a match out of unrelated slots.
        var client = _factory.CreateClient();

        var marker = Guid.NewGuid().ToString("N");
        await client.PostAsJsonAsync("/v1/chat/completions", new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"First unique prompt ({marker}-A)" } },
            MaxTokens = 2,
        }, TestContext.Current.CancellationToken);

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Completely different query ({marker}-B)" } },
            MaxTokens = 2,
        }, TestContext.Current.CancellationToken);

        resp.EnsureSuccessStatusCode();
        int cached = int.Parse(resp.Headers.GetValues("X-Cached-Tokens").First());
        // The chat template wraps user messages with role markers like
        // "<|im_start|>user\n...", so the VERY first few tokens (BOS,
        // role-prefix special tokens) may match between any two user-only
        // prompts regardless of content. We tolerate a small fixed tail
        // but would flag a match of dozens of tokens as "not really
        // different content."
        Assert.True(cached < 10,
            $"Unrelated prompts should not share a long cached prefix; got {cached} cached tokens.");
    }

    [Fact]
    public async Task Output_Is_Consistent_Across_Cache_Hits()
    {
        // Correctness guard: a prompt-cached second request must produce
        // the same output as a cold one under greedy sampling. If it
        // doesn't, firstNewIndex is wrong or the slot's KV state doesn't
        // match what we're claiming to cache.
        var client = _factory.CreateClient();
        var marker = Guid.NewGuid().ToString("N");
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Say one short sentence ({marker})." } },
            MaxTokens = 10,
            Temperature = 0.0f,
        };

        var r1 = await client.PostAsJsonAsync("/v1/chat/completions", req, TestContext.Current.CancellationToken);
        r1.EnsureSuccessStatusCode();
        var b1 = await r1.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        var r2 = await client.PostAsJsonAsync("/v1/chat/completions", req, TestContext.Current.CancellationToken);
        r2.EnsureSuccessStatusCode();
        var b2 = await r2.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(b1!.Choices[0].Message.Content, b2!.Choices[0].Message.Content);
        Assert.True(int.Parse(r2.Headers.GetValues("X-Cached-Tokens").First()) > 0,
            "Sanity: second request should have used the cache.");
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
