using System.Net;
using System.Net.Http.Headers;
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
        Assert.False(string.IsNullOrWhiteSpace(body.Choices[0].Message.Content?.Text ?? ""),
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
        var assistantReply = body1!.Choices[0].Message.Content?.Text ?? "";
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

        Assert.Equal(b1!.Choices[0].Message.Content?.Text ?? "", b2!.Choices[0].Message.Content?.Text ?? "");
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
            return body!.Choices[0].Message.Content?.Text ?? "";
        }

        var results = await Task.WhenAll(
            Run("Name a colour."),
            Run("Name a fruit."));

        Assert.False(string.IsNullOrWhiteSpace(results[0]));
        Assert.False(string.IsNullOrWhiteSpace(results[1]));
    }

    // ----- /slots observability -----

    [Fact]
    public async Task Slots_Reports_Pool_Size_And_Shape()
    {
        var client = _factory.CreateClient();
        var slots = await client.GetFromJsonAsync<LlamaCpp.Bindings.Server.Services.SlotSnapshot[]>(
            "/slots", TestContext.Current.CancellationToken);
        Assert.NotNull(slots);
        // Factory configures MaxSequenceCount=2, so the pool has 2 slots.
        Assert.Equal(2, slots!.Length);
        Assert.Equal(0, slots[0].SlotIndex);
        Assert.Equal(1, slots[1].SlotIndex);
        // Sequence ids must be distinct within the pool.
        Assert.NotEqual(slots[0].SequenceId, slots[1].SequenceId);
    }

    [Fact]
    public async Task Slots_Show_CachedTokens_After_A_Completion()
    {
        var client = _factory.CreateClient();
        var marker = Guid.NewGuid().ToString("N");
        await client.PostAsJsonAsync("/v1/chat/completions", new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"slots-marker-{marker}" } },
            MaxTokens = 3,
        }, TestContext.Current.CancellationToken);

        var slots = await client.GetFromJsonAsync<LlamaCpp.Bindings.Server.Services.SlotSnapshot[]>(
            "/slots", TestContext.Current.CancellationToken);
        Assert.NotNull(slots);
        Assert.All(slots!, s => Assert.False(s.InUse));
        // At least one slot should hold some cached tokens after a request.
        Assert.Contains(slots, s => s.CachedTokenCount > 0);
    }

    // ----- logit_bias -----

    [Fact]
    public async Task LogitBias_Rejects_Non_Numeric_Key()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = "hello" } },
            MaxTokens = 2,
            LogitBias = new() { ["not-a-number"] = -100f },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task LogitBias_Rejects_Out_Of_Range_Token()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = "hello" } },
            MaxTokens = 2,
            LogitBias = new() { ["99999999"] = -100f },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task LogitBias_Changes_Output_Tokens()
    {
        // Greedy baseline run collects the first generated text. Then we
        // repeat with a -100 bias on the first tokens of that output — the
        // model is forced to pick something else. Proves the bias actually
        // reaches the sampler; a no-op implementation would produce the
        // same text twice.
        var client = _factory.CreateClient();
        var marker = Guid.NewGuid().ToString("N");
        var baseReq = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Say 'hi'. ({marker})" } },
            MaxTokens = 3,
            Temperature = 0.0f,
        };
        var baseResp = await client.PostAsJsonAsync(
            "/v1/chat/completions", baseReq, TestContext.Current.CancellationToken);
        baseResp.EnsureSuccessStatusCode();
        var baseline = await baseResp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        var baseText = baseline!.Choices[0].Message.Content?.Text ?? "";
        Assert.False(string.IsNullOrEmpty(baseText), "baseline produced empty output");

        // Tokenize the baseline output via the /completion endpoint's vocab
        // indirectly — we don't expose a tokenize endpoint, but we can grab
        // any token id in the model's range by just banning a huge chunk
        // below 1024. Pick token ids in the common ASCII / punctuation range
        // that are near-guaranteed to appear in the first few generated
        // tokens of a "hi"-style reply.
        var banned = new Dictionary<string, float>();
        for (int i = 0; i < 2000; i++) banned[i.ToString()] = -100f;

        var biasedReq = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Say 'hi'. ({marker})" } },
            MaxTokens = 3,
            Temperature = 0.0f,
            LogitBias = banned,
        };
        var biasedResp = await client.PostAsJsonAsync(
            "/v1/chat/completions", biasedReq, TestContext.Current.CancellationToken);
        biasedResp.EnsureSuccessStatusCode();
        var biased = await biasedResp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.NotEqual(baseText, biased!.Choices[0].Message.Content?.Text ?? "");
    }

    // ----- Extended sampling knobs (min_p, typical_p, top_n_sigma, xtc, dry, mirostat, penalties) -----

    [Fact]
    public async Task MinP_And_TypicalP_And_TopNSigma_Are_Accepted()
    {
        // These all sit in the truncation stages of the sampler chain.
        // Semantic verification for each is hard to write as a stable
        // test (they shape the distribution in subtle ways), so we settle
        // for "request completes without error and returns non-empty
        // output" as the signal that the chain actually built.
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"short reply {Guid.NewGuid():N}" } },
            MaxTokens = 4,
            Temperature = 0.7f,
            MinP = 0.05f,
            TypicalP = 0.95f,
            TopNSigma = 1.5f,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.False(string.IsNullOrWhiteSpace(body!.Choices[0].Message.Content?.Text ?? ""));
    }

    [Fact]
    public async Task Heavy_Repeat_Penalty_Changes_Output_Vs_Baseline()
    {
        // Baseline greedy vs. greedy + repeat_penalty=2.0. A large penalty
        // forces a different token path as soon as any prompt token would
        // have been a candidate for repetition. This catches a broken
        // wiring where the penalties stage silently gets dropped from the
        // chain — the two runs would otherwise be identical.
        var client = _factory.CreateClient();
        var marker = Guid.NewGuid().ToString("N");
        var prompt = $"Repeat: the the the the the the ({marker})";

        var baseline = await PostChat(client, prompt, r => r.Temperature = 0.0f);
        var penalised = await PostChat(client, prompt, r =>
        {
            r.Temperature = 0.0f;
            r.RepeatPenalty = 2.0f;
            r.RepeatLastN = 64;
        });

        Assert.False(string.IsNullOrWhiteSpace(baseline));
        Assert.False(string.IsNullOrWhiteSpace(penalised));
        Assert.NotEqual(baseline, penalised);
    }

    [Fact]
    public async Task Mirostat_V2_Produces_Output_And_Overrides_Temperature()
    {
        // Mirostat terminal replaces temperature + truncation. Smoke: the
        // request succeeds with non-empty output even though temperature
        // would normally be required for stochastic sampling. Any throw
        // from the sampler builder (e.g. a missing terminal) fails hard.
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Short. {Guid.NewGuid():N}" } },
            MaxTokens = 6,
            Mirostat = 2,
            MirostatTau = 5.0f,
            MirostatEta = 0.1f,
            // Deliberately leave Temperature / TopK / TopP unset — mirostat
            // must supply its own terminal.
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.False(string.IsNullOrWhiteSpace(body!.Choices[0].Message.Content?.Text ?? ""));
    }

    [Fact]
    public async Task Invalid_Mirostat_Value_Rejected_With_400()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = "hi" } },
            MaxTokens = 2,
            Mirostat = 7, // only 0/1/2 valid
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Dry_Sampler_Accepts_Custom_Sequence_Breakers()
    {
        // DRY has the most moving parts of any sampler knob (multiplier,
        // base, allowed_length, penalty_last_n, sequence_breakers). The
        // lift here isn't "does it penalise" — that's a binding-level
        // test — it's "does the server marshal all five fields through
        // the chain without tripping over the list-of-strings breakers."
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"short. {Guid.NewGuid():N}" } },
            MaxTokens = 4,
            Temperature = 0.0f,
            DryMultiplier = 0.8f,
            DryBase = 1.75f,
            DryAllowedLength = 2,
            DryPenaltyLastN = 64,
            DrySequenceBreakers = new() { "\n", ".", "," },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.False(string.IsNullOrWhiteSpace(body!.Choices[0].Message.Content?.Text ?? ""));
    }

    private async Task<string> PostChat(
        HttpClient client, string userContent, Action<ChatCompletionsRequest> mutate)
    {
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = userContent } },
            MaxTokens = 12,
        };
        mutate(req);
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        return body!.Choices[0].Message.Content?.Text ?? "";
    }

    // ----- stop sequences -----

    [Fact]
    public async Task Stop_Array_Accepts_Multiple_And_First_Hit_Wins()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Count: 1 2 3 4 5 6 ({Guid.NewGuid():N})" } },
            MaxTokens = 30,
            Temperature = 0.0f,
            Stop = JsonDocument.Parse("""["\"4\"", " 4 ", "4 5"]""").RootElement,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        var text = body!.Choices[0].Message.Content?.Text ?? "";

        // Whichever stop fires, none of the three should appear in the
        // returned text — the stop is always stripped.
        Assert.DoesNotContain(" 4 ", text);
        Assert.DoesNotContain("4 5", text);
    }

    [Fact]
    public async Task Stop_Rejects_Non_String_Array_Entries_With_400()
    {
        var client = _factory.CreateClient();
        // Array with a number entry — not valid per our parse rules.
        var raw = """
            {"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stop":["ok",123]}
            """;
        using var content = new StringContent(raw, System.Text.Encoding.UTF8, "application/json");
        var resp = await client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Stop_String_Truncates_Output_And_Sets_Finish_Reason()
    {
        // Generation should halt the moment the emitted text ends with
        // the stop string, and the stop itself must be absent from the
        // returned content. We pick a prompt that's very likely to emit
        // "Human" early (a classic chat-style follow-up).
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Continue: 'A B C D E F G H'. Then stop. ({Guid.NewGuid():N})" } },
            MaxTokens = 40,
            Temperature = 0.0f,
            Stop = JsonDocument.Parse("\" C \"").RootElement, // stop after "A B"
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.NotNull(body);
        var text = body!.Choices[0].Message.Content?.Text ?? "";
        Assert.DoesNotContain(" C ", text);
        // The model should at minimum emit something before the stop fires;
        // empty output means the first token was already the stop (odd but
        // legal) — either way finish_reason must report "stop".
        Assert.Equal("stop", body.Choices[0].FinishReason);
    }

    [Fact]
    public async Task Streaming_With_Stop_Emits_Stop_Finish_Reason_In_Final_Chunk()
    {
        var client = _factory.CreateClient();
        // Stop on a single period — guaranteed to appear in any
        // multi-sentence answer. The test isn't checking "does the model
        // echo a specific phrase" (that's flaky on GPU because cached KV
        // shifts the argmax) — only that the stop hooks into the
        // streaming path and reports finish_reason="stop".
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Write two short sentences about rain. ({Guid.NewGuid():N})" } },
            MaxTokens = 80,
            Temperature = 0.0f,
            Stream = true,
            Stop = JsonDocument.Parse("\".\"").RootElement,
        };

        using var httpReq = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
        {
            Content = JsonContent.Create(req),
        };
        using var resp = await client.SendAsync(
            httpReq, HttpCompletionOption.ResponseHeadersRead, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.OK, resp.StatusCode);

        using var stream = await resp.Content.ReadAsStreamAsync(TestContext.Current.CancellationToken);
        using var reader = new StreamReader(stream);

        var accumulated = new StringBuilder();
        string? finalFinishReason = null;

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync(TestContext.Current.CancellationToken);
            if (line is null) break;
            if (!line.StartsWith("data: ")) continue;
            var payload = line["data: ".Length..];
            if (payload == "[DONE]") break;

            using var doc = JsonDocument.Parse(payload);
            var choice = doc.RootElement.GetProperty("choices")[0];
            if (choice.GetProperty("delta").TryGetProperty("content", out var c) &&
                c.ValueKind == JsonValueKind.String)
            {
                accumulated.Append(c.GetString());
            }
            if (choice.TryGetProperty("finish_reason", out var fr) &&
                fr.ValueKind == JsonValueKind.String)
            {
                finalFinishReason = fr.GetString();
            }
        }

        Assert.Equal("stop", finalFinishReason);
        Assert.DoesNotContain(".", accumulated.ToString());
    }

    [Fact]
    public async Task Completion_Endpoint_Accepts_Stop_Field_Without_Error()
    {
        // Spot-check that /completion — not just /v1/chat/completions —
        // actually parses + wires the stop field through to the
        // StopMatcher. We don't assert finish_reason here because the
        // raw /completion path (no chat template) is especially
        // sensitive to cached-KV-state variance on GPU: the model's
        // exact output for a given prompt differs between runs, so a
        // content-matched stop is inherently flaky.
        //
        // Semantic coverage for stops lives in:
        //   - StopMatcherTests (model-free unit tests, all nine cases)
        //   - Stop_String_Truncates_Output_And_Sets_Finish_Reason
        //     (chat endpoint, proves the end-to-end match path)
        //   - Stop_Array_Accepts_Multiple_And_First_Hit_Wins (chat,
        //     proves array parsing + per-entry matching)
        var client = _factory.CreateClient();
        var req = new CompletionRequest
        {
            Prompt = $"Once upon a time, in a distant land ({Guid.NewGuid():N}) ",
            MaxTokens = 32,
            Temperature = 0.0f,
            Stop = JsonDocument.Parse("""[".", ",", "\n", " "]""").RootElement,
        };
        var resp = await client.PostAsJsonAsync("/completion", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<CompletionResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotNull(body);
        // Either "stop" or "length" is a valid outcome here; we only
        // guard against 4xx/5xx and malformed responses.
        Assert.Contains(body!.StopReason, new[] { "stop", "length" });
    }

    // ----- Multimodal content part handling -----

    [Fact]
    public async Task Chat_Accepts_Array_Content_With_Only_Text_Parts()
    {
        // Text-only multipart content (no images) must still work even
        // when MmprojHost isn't configured — it should flatten to a
        // plain string and take the normal text-only path.
        var client = _factory.CreateClient();
        string marker = Guid.NewGuid().ToString("N");
        string raw =
            "{\"messages\":[{\"role\":\"user\",\"content\":[" +
            "{\"type\":\"text\",\"text\":\"Hello, \"}," +
            "{\"type\":\"text\",\"text\":\"world. " + marker + "\"}" +
            "]}],\"max_tokens\":4}";
        using var content = new StringContent(raw, System.Text.Encoding.UTF8, "application/json");
        var resp = await client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.False(string.IsNullOrWhiteSpace(body!.Choices[0].Message.Content?.Text ?? ""));
    }

    [Fact]
    public async Task Chat_Rejects_Image_Part_With_400_When_Mmproj_Not_Configured()
    {
        // Default fixture has no MmprojPath set. An image_url part should
        // get a 400 before any generation happens — tells the caller the
        // feature is disabled for this deployment.
        var client = _factory.CreateClient();
        const string TinyPng = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        string raw =
            "{\"messages\":[{\"role\":\"user\",\"content\":[" +
            "{\"type\":\"text\",\"text\":\"Describe.\"}," +
            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + TinyPng + "\"}}" +
            "]}],\"max_tokens\":4}";
        using var content = new StringContent(raw, System.Text.Encoding.UTF8, "application/json");
        var resp = await client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_Rejects_Unknown_Content_Part_Type_With_400()
    {
        var client = _factory.CreateClient();
        var raw = """
            {
              "messages": [
                {"role": "user", "content": [
                   {"type": "video_url", "video_url": {"url": "http://x"}}
                ]}
              ],
              "max_tokens": 2
            }
            """;
        using var content = new StringContent(raw, System.Text.Encoding.UTF8, "application/json");
        var resp = await client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_Rejects_Non_Data_Url_Image_With_400()
    {
        var client = _factory.CreateClient();
        var raw = """
            {
              "messages": [
                {"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
                ]}
              ],
              "max_tokens": 2
            }
            """;
        using var content = new StringContent(raw, System.Text.Encoding.UTF8, "application/json");
        var resp = await client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    // ----- Per-request timings + /metrics -----

    [Fact]
    public async Task Chat_Response_Includes_Timings_Block()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"hi {Guid.NewGuid():N}" } },
            MaxTokens = 4,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotNull(body);
        Assert.NotNull(body!.Timings);
        Assert.True(body.Timings!.PromptN > 0, "prompt_n should be positive");
        Assert.True(body.Timings.PredictedN > 0, "predicted_n should be positive");
        Assert.True(body.Timings.PredictedMs >= 0, "predicted_ms should be non-negative");
    }

    [Fact]
    public async Task Metrics_Endpoint_Returns_Prometheus_Text()
    {
        // Fire a request to seed some counters, then scrape /metrics.
        var client = _factory.CreateClient();
        await client.PostAsJsonAsync("/v1/chat/completions", new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"hi {Guid.NewGuid():N}" } },
            MaxTokens = 2,
        }, TestContext.Current.CancellationToken);

        var resp = await client.GetAsync("/metrics", TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        Assert.StartsWith("text/plain", resp.Content.Headers.ContentType?.MediaType ?? "");

        var text = await resp.Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        // Spot-check the canonical counter names — the test shouldn't
        // pin exact values (they change with every test run) but the
        // presence of the lines confirms the Prometheus format and the
        // counter pipeline are both wired.
        Assert.Contains("llama_requests_total{", text);
        Assert.Contains("llama_tokens_generated_total", text);
        Assert.Contains("llama_tokens_prompt_total", text);
        Assert.Contains("llama_slot_in_use{", text);
        Assert.Contains("llama_slot_cached_tokens{", text);
        // HELP/TYPE comments are required for proper Prometheus scraping.
        Assert.Contains("# HELP llama_requests_total", text);
        Assert.Contains("# TYPE llama_requests_total counter", text);
    }

    [Fact]
    public async Task Metrics_Requests_Total_Counts_Each_Request()
    {
        var client = _factory.CreateClient();
        // Pull starting value (request itself increments /metrics count too —
        // we pick a specific endpoint to watch: /health, which we control).
        await client.GetAsync("/health", TestContext.Current.CancellationToken);
        var scrape1 = await (await client.GetAsync("/metrics", TestContext.Current.CancellationToken))
            .Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        int before = CountHealthRequests(scrape1);

        // Fire three more health requests and re-scrape.
        for (int i = 0; i < 3; i++)
        {
            await client.GetAsync("/health", TestContext.Current.CancellationToken);
        }
        var scrape2 = await (await client.GetAsync("/metrics", TestContext.Current.CancellationToken))
            .Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        int after = CountHealthRequests(scrape2);

        Assert.True(after >= before + 3,
            $"expected at least 3 more /health requests; before={before}, after={after}");
    }

    private static int CountHealthRequests(string promText)
    {
        // Parse the single line matching llama_requests_total{endpoint="/health",status="200"} N
        foreach (var line in promText.Split('\n'))
        {
            if (line.StartsWith("llama_requests_total{") &&
                line.Contains("endpoint=\"/health\"") &&
                line.Contains("status=\"200\""))
            {
                var space = line.LastIndexOf(' ');
                if (space > 0 && int.TryParse(line[(space + 1)..], out var n)) return n;
            }
        }
        return 0;
    }

    // ----- CORS (default fixture: CORS disabled) -----

    [Fact]
    public async Task CorsHeaders_Absent_When_Not_Configured()
    {
        // Baseline: the default fixture doesn't set CorsAllowedOrigins.
        // The middleware chain therefore shouldn't register CORS, and
        // responses shouldn't carry Access-Control-Allow-Origin even
        // when the client includes an Origin header.
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/health");
        req.Headers.Add("Origin", "https://example.com");
        using var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        Assert.False(resp.Headers.Contains("Access-Control-Allow-Origin"),
            "CORS header should not be set when CorsAllowedOrigins is empty.");
    }

    [Fact]
    public async Task Cancelled_Stream_Releases_Pool_Slot()
    {
        // Streaming requests are the path where a misbehaving
        // RequestAborted propagation would leak slots — a dropped client
        // mid-stream should cancel the generator and release its session.
        // We simulate the drop by cancelling the client's own CTS partway
        // through the SSE stream, then poll /slots and expect no slot to
        // still be marked InUse.
        var client = _factory.CreateClient();
        using var cts = new CancellationTokenSource();

        using var httpReq = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
        {
            Content = JsonContent.Create(new ChatCompletionsRequest
            {
                Messages = new() { new() { Role = "user", Content = $"Count slowly. ({Guid.NewGuid():N})" } },
                MaxTokens = 64,
                Stream = true,
            }),
        };

        var streamTask = client.SendAsync(
            httpReq, HttpCompletionOption.ResponseHeadersRead, cts.Token);
        using var resp = await streamTask;
        Assert.Equal(System.Net.HttpStatusCode.OK, resp.StatusCode);

        using (var stream = await resp.Content.ReadAsStreamAsync(TestContext.Current.CancellationToken))
        using (var reader = new StreamReader(stream))
        {
            // Read a couple of chunks to prove the stream has actually
            // started producing, then cancel.
            int dataLines = 0;
            while (dataLines < 2)
            {
                var line = await reader.ReadLineAsync(TestContext.Current.CancellationToken);
                if (line is null) break;
                if (line.StartsWith("data: ")) dataLines++;
            }
            cts.Cancel();
        }

        // Poll /slots until InUse drops to false across all slots — give
        // the server a generous window (generation loop observes the
        // cancellation on the next iteration, plus lease dispose takes a
        // moment). If cancellation isn't propagated this will time out.
        var deadline = DateTimeOffset.UtcNow.AddSeconds(10);
        while (DateTimeOffset.UtcNow < deadline)
        {
            var slots = await client.GetFromJsonAsync<LlamaCpp.Bindings.Server.Services.SlotSnapshot[]>(
                "/slots", TestContext.Current.CancellationToken);
            if (slots!.All(s => !s.InUse))
            {
                return; // success
            }
            await Task.Delay(100, TestContext.Current.CancellationToken);
        }
        Assert.Fail("A slot remained marked InUse more than 10 seconds after the client cancelled — pool leak.");
    }

    // ----- Structured output (grammar / response_format / json_schema) -----

    [Fact]
    public async Task ResponseFormat_JsonObject_Produces_Parseable_Json()
    {
        // "json_object" mode attaches the bundled JSON grammar (any valid
        // JSON). The output must parse as JSON — nothing more, but
        // nothing less either.
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Return a JSON object with a name and age. ({Guid.NewGuid():N})" } },
            MaxTokens = 64,
            Temperature = 0.0f,
            ResponseFormat = new ResponseFormat { Type = "json_object" },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        var content = body!.Choices[0].Message.Content?.Text ?? "".Trim();
        Assert.False(string.IsNullOrEmpty(content));
        // Must parse as JSON. An ungrammared model would wrap the object
        // in natural-language prose like "Here's your JSON: {...}" which
        // fails this parse.
        using var doc = JsonDocument.Parse(content);
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    [Fact]
    public async Task ResponseFormat_JsonSchema_Constrains_Output_To_Schema()
    {
        // Schema requires an object with a "name" string and "age" int.
        // A well-formed response should parse AND have both fields of
        // the right types.
        var client = _factory.CreateClient();
        var schemaJson = """
            {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "age":  { "type": "integer" }
              },
              "required": ["name", "age"]
            }
            """;
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Make up a person. ({Guid.NewGuid():N})" } },
            MaxTokens = 64,
            Temperature = 0.0f,
            ResponseFormat = new ResponseFormat
            {
                Type = "json_schema",
                JsonSchema = new JsonSchemaSpec
                {
                    Name = "Person",
                    Schema = JsonDocument.Parse(schemaJson).RootElement,
                },
            },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        var content = body!.Choices[0].Message.Content?.Text ?? "".Trim();
        using var doc = JsonDocument.Parse(content);
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
        Assert.True(doc.RootElement.TryGetProperty("name", out var name));
        Assert.Equal(JsonValueKind.String, name.ValueKind);
        Assert.True(doc.RootElement.TryGetProperty("age", out var age));
        Assert.Equal(JsonValueKind.Number, age.ValueKind);
    }

    [Fact]
    public async Task Raw_Grammar_Field_Constrains_Output()
    {
        // Very narrow grammar: output must be exactly one of three words.
        // Proves the raw `grammar` passthrough actually reaches the
        // sampler — any alternative path (ignored field, silent drop)
        // would let the model say whatever it wanted.
        var client = _factory.CreateClient();
        const string Gbnf = """
            root ::= "yes" | "no" | "maybe"
            """;
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Answer the question. ({Guid.NewGuid():N})" } },
            MaxTokens = 8,
            Temperature = 0.0f,
            Grammar = Gbnf,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        var content = body!.Choices[0].Message.Content?.Text ?? "".Trim();
        Assert.Contains(content, new[] { "yes", "no", "maybe" });
    }

    [Fact]
    public async Task JsonSchema_Short_Form_Compiles_Like_Response_Format()
    {
        // llama-server's bare `json_schema` field should compile via the
        // same code path. Minimal schema = any object.
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Emit JSON. ({Guid.NewGuid():N})" } },
            MaxTokens = 32,
            Temperature = 0.0f,
            JsonSchemaShort = JsonDocument.Parse("""{"type":"object"}""").RootElement,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        var content = body!.Choices[0].Message.Content?.Text ?? "".Trim();
        using var doc = JsonDocument.Parse(content);
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    [Fact]
    public async Task Unknown_ResponseFormat_Type_Returns_400()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = "hi" } },
            MaxTokens = 4,
            ResponseFormat = new ResponseFormat { Type = "xml" },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task JsonSchema_Without_Schema_Field_Returns_400()
    {
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = "hi" } },
            MaxTokens = 4,
            ResponseFormat = new ResponseFormat
            {
                Type = "json_schema",
                JsonSchema = new JsonSchemaSpec { Name = "empty" }, // no Schema
            },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Raw_Grammar_Wins_Over_Json_Schema_And_Response_Format()
    {
        // Precedence test: grammar > json_schema > response_format. We
        // set all three; the raw grammar restricts output to "yes", so
        // the response should be a plain word, not JSON.
        var client = _factory.CreateClient();
        var req = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = $"Answer. ({Guid.NewGuid():N})" } },
            MaxTokens = 8,
            Temperature = 0.0f,
            Grammar = """root ::= "yes" """,
            JsonSchemaShort = JsonDocument.Parse("""{"type":"object"}""").RootElement,
            ResponseFormat = new ResponseFormat { Type = "json_object" },
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ChatCompletionsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("yes", body!.Choices[0].Message.Content?.Text ?? "".Trim());
    }

    // ----- /v1/embeddings: 501 when no embedding model is configured -----

    [Fact]
    public async Task V1Embeddings_Returns_501_When_Not_Configured()
    {
        // The default fixture runs without an embedding model, so the
        // endpoint should register but respond with 501 Not Implemented.
        // That's more informative than a 404 — tells the caller the feature
        // exists but is disabled for this deployment.
        var client = _factory.CreateClient();
        var body = new EmbeddingsRequest
        {
            Input = JsonDocument.Parse("\"hello\"").RootElement,
            Model = "any",
        };
        var resp = await client.PostAsJsonAsync("/v1/embeddings", body, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.NotImplemented, resp.StatusCode);
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

/// <summary>
/// API-key auth tests. Uses a dedicated factory with two configured keys
/// (inline + file) so every auth path has coverage.
/// </summary>
public class LlamaServerAuthTests : IClassFixture<LlamaServerAuthTests.AuthFactory>
{
    private readonly AuthFactory _factory;
    public LlamaServerAuthTests(AuthFactory factory) => _factory = factory;

    private const string GoodKey   = "test-key-inline-12345";
    private const string GoodKey2  = "test-key-from-file-abcde";
    private const string WrongKey  = "definitely-not-the-key-xyz";

    [Fact]
    public async Task Health_Is_Always_Open_Even_With_Keys_Configured()
    {
        // Liveness probes must not need credentials — container orchestrators
        // can't be expected to ship API keys into every probe config.
        var client = _factory.CreateClient();
        var resp = await client.GetAsync("/health", TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.OK, resp.StatusCode);
    }

    [Fact]
    public async Task Protected_Endpoint_Returns_401_Without_Key()
    {
        var client = _factory.CreateClient();
        var resp = await client.GetAsync("/v1/models", TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    [Fact]
    public async Task Protected_Endpoint_Returns_401_With_Wrong_Key()
    {
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", WrongKey);
        var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    [Fact]
    public async Task Bearer_Token_With_Inline_Key_Is_Accepted()
    {
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", GoodKey);
        var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.OK, resp.StatusCode);
    }

    [Fact]
    public async Task Bearer_Token_With_File_Key_Is_Accepted()
    {
        // The AuthFactory writes GoodKey2 into a temporary file and points
        // ApiKeyFile at it. This verifies the file-loading path actually
        // merges its entries into the accepted set.
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", GoodKey2);
        var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.OK, resp.StatusCode);
    }

    [Fact]
    public async Task XApiKey_Header_Is_Accepted_As_Fallback()
    {
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Add("X-Api-Key", GoodKey);
        var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.OK, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_Completions_Requires_Auth()
    {
        // Spot-check that auth gates the hot path, not just /v1/models.
        var client = _factory.CreateClient();
        var body = new ChatCompletionsRequest
        {
            Messages = new() { new() { Role = "user", Content = "hi" } },
            MaxTokens = 2,
        };
        var resp = await client.PostAsJsonAsync(
            "/v1/chat/completions", body, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    public sealed class AuthFactory : WebApplicationFactory<Program>, IDisposable
    {
        private readonly string _keyFile;

        public AuthFactory()
        {
            _keyFile = Path.Combine(Path.GetTempPath(), $"llama-server-auth-{Guid.NewGuid():N}.keys");
            File.WriteAllLines(_keyFile, new[]
            {
                "# comment line — ignored",
                "",
                GoodKey2,
            });
        }

        protected override IHost CreateHost(IHostBuilder builder)
        {
            var modelPath = TestModelProvider.EnsureModelPath();
            builder.ConfigureAppConfiguration(cfg =>
            {
                cfg.AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["LlamaServer:ModelPath"]         = modelPath,
                    ["LlamaServer:ContextSize"]       = "512",
                    ["LlamaServer:MaxSequenceCount"]  = "1",
                    ["LlamaServer:GpuLayerCount"]     = "0", // CPU — keeps the second model light
                    ["LlamaServer:OffloadKqv"]        = "false",
                    ["LlamaServer:MaxOutputTokens"]   = "16",
                    ["LlamaServer:Urls"]              = "",
                    ["LlamaServer:ApiKeys:0"]         = GoodKey,
                    ["LlamaServer:ApiKeyFile"]        = _keyFile,
                });
            });
            return base.CreateHost(builder);
        }

        protected override void Dispose(bool disposing)
        {
            try
            {
                if (File.Exists(_keyFile)) File.Delete(_keyFile);
            }
            catch { /* best-effort cleanup */ }
            base.Dispose(disposing);
        }
    }
}

/// <summary>
/// Tests for CORS behaviour when the server is configured with an
/// origin allow-list. Uses a tiny CPU-only factory so the second
/// server is cheap to spin up.
/// </summary>
public class LlamaServerCorsTests : IClassFixture<LlamaServerCorsTests.CorsFactory>
{
    private readonly CorsFactory _factory;
    public LlamaServerCorsTests(CorsFactory factory) => _factory = factory;

    private const string AllowedOrigin = "https://app.example.com";
    private const string BlockedOrigin = "https://evil.example.com";

    [Fact]
    public async Task Allowed_Origin_Gets_AccessControl_Header()
    {
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/health");
        req.Headers.Add("Origin", AllowedOrigin);
        using var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        Assert.True(resp.Headers.TryGetValues("Access-Control-Allow-Origin", out var vals));
        Assert.Equal(AllowedOrigin, vals.First());
    }

    [Fact]
    public async Task Blocked_Origin_Does_Not_Get_AccessControl_Header()
    {
        // Non-allow-listed Origin: request still succeeds (CORS policing
        // is the browser's job) but the Access-Control-Allow-Origin
        // header is absent, which is what tells the browser to refuse.
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Get, "/health");
        req.Headers.Add("Origin", BlockedOrigin);
        using var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        Assert.False(resp.Headers.Contains("Access-Control-Allow-Origin"));
    }

    [Fact]
    public async Task Preflight_Options_Returns_AccessControl_Headers()
    {
        // Browsers send an OPTIONS preflight before any "complex" CORS
        // request (POST with a non-simple Content-Type, custom headers
        // like Authorization). The preflight must succeed without hitting
        // the endpoint handler or the real POST never happens.
        var client = _factory.CreateClient();
        using var req = new HttpRequestMessage(HttpMethod.Options, "/v1/chat/completions");
        req.Headers.Add("Origin", AllowedOrigin);
        req.Headers.Add("Access-Control-Request-Method", "POST");
        req.Headers.Add("Access-Control-Request-Headers", "authorization,content-type");
        using var resp = await client.SendAsync(req, TestContext.Current.CancellationToken);

        Assert.True(
            resp.StatusCode == System.Net.HttpStatusCode.OK ||
            resp.StatusCode == System.Net.HttpStatusCode.NoContent,
            $"preflight status was {resp.StatusCode}; expected 200 or 204");
        Assert.True(resp.Headers.TryGetValues("Access-Control-Allow-Origin", out var origin));
        Assert.Equal(AllowedOrigin, origin.First());
        Assert.True(resp.Headers.Contains("Access-Control-Allow-Methods"));
    }

    public sealed class CorsFactory : WebApplicationFactory<Program>
    {
        protected override IHost CreateHost(IHostBuilder builder)
        {
            var modelPath = TestModelProvider.EnsureModelPath();
            builder.ConfigureAppConfiguration(cfg =>
            {
                cfg.AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["LlamaServer:ModelPath"]            = modelPath,
                    ["LlamaServer:ContextSize"]          = "512",
                    ["LlamaServer:MaxSequenceCount"]     = "1",
                    ["LlamaServer:GpuLayerCount"]        = "0",
                    ["LlamaServer:OffloadKqv"]           = "false",
                    ["LlamaServer:MaxOutputTokens"]      = "8",
                    ["LlamaServer:Urls"]                 = "",
                    ["LlamaServer:CorsAllowedOrigins:0"] = AllowedOrigin,
                });
            });
            return base.CreateHost(builder);
        }
    }
}

/// <summary>
/// End-to-end tests for <c>/v1/embeddings</c>. Uses its own factory so a
/// dedicated embedding model is loaded — nomic-embed-text-v1.5 by
/// default, auto-fetched on first use. When the download fails (offline)
/// every test in this class skips gracefully.
/// </summary>
public class LlamaEmbeddingsTests : IClassFixture<LlamaEmbeddingsTests.EmbedFactory>
{
    private readonly EmbedFactory _factory;
    public LlamaEmbeddingsTests(EmbedFactory factory) => _factory = factory;

    [Fact]
    public async Task Single_Input_Returns_Embedding_Vector()
    {
        if (!_factory.EmbedModelAvailable) Assert.Skip("nomic-embed GGUF unavailable; set LLAMACPP_TEST_EMBEDDING_MODEL or allow auto-download.");

        var client = _factory.CreateClient();
        var body = new EmbeddingsRequest
        {
            Input = JsonDocument.Parse("\"hello, world\"").RootElement,
        };
        var resp = await client.PostAsJsonAsync("/v1/embeddings", body, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();

        var parsed = await resp.Content.ReadFromJsonAsync<EmbeddingsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotNull(parsed);
        Assert.Equal("list", parsed!.Object);
        Assert.Single(parsed.Data);
        Assert.Equal(0, parsed.Data[0].Index);
        Assert.NotEmpty(parsed.Data[0].Embedding);

        // nomic-embed-text-v1.5 returns 768-dim vectors. Any other dim is
        // either the wrong model loaded or a silent truncation somewhere.
        Assert.Equal(768, parsed.Data[0].Embedding.Length);

        // Vector should not be all-zero — an obvious failure mode when the
        // decode path silently short-circuits. Any real forward pass
        // produces a vector with non-trivial magnitude.
        double magnitude = Math.Sqrt(parsed.Data[0].Embedding.Sum(x => (double)x * x));
        Assert.True(magnitude > 0.01, $"embedding magnitude {magnitude} is too small to be real");

        // Usage should report at least one prompt token.
        Assert.True(parsed.Usage.PromptTokens > 0);
        Assert.Equal(parsed.Usage.PromptTokens, parsed.Usage.TotalTokens);
    }

    [Fact]
    public async Task Array_Input_Returns_One_Embedding_Per_Entry_With_Correct_Index()
    {
        if (!_factory.EmbedModelAvailable) Assert.Skip("nomic-embed GGUF unavailable; set LLAMACPP_TEST_EMBEDDING_MODEL or allow auto-download.");

        var client = _factory.CreateClient();
        var body = new EmbeddingsRequest
        {
            Input = JsonDocument.Parse("""["alpha","beta","gamma"]""").RootElement,
        };
        var resp = await client.PostAsJsonAsync("/v1/embeddings", body, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();

        var parsed = await resp.Content.ReadFromJsonAsync<EmbeddingsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotNull(parsed);
        Assert.Equal(3, parsed!.Data.Count);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(i, parsed.Data[i].Index);
            Assert.Equal(768, parsed.Data[i].Embedding.Length);
        }
        // Three distinct inputs → three distinct vectors. If two come back
        // identical the endpoint is caching improperly across inputs.
        Assert.NotEqual(parsed.Data[0].Embedding, parsed.Data[1].Embedding);
        Assert.NotEqual(parsed.Data[1].Embedding, parsed.Data[2].Embedding);
    }

    [Fact]
    public async Task Semantic_Similarity_Is_Higher_For_Related_Texts()
    {
        // Sanity: embeddings should put "cat" and "kitten" closer than
        // "cat" and "spaceship". A broken pooling head produces vectors
        // whose cosine similarity is essentially random across inputs,
        // which this test catches.
        if (!_factory.EmbedModelAvailable) Assert.Skip("nomic-embed GGUF unavailable; set LLAMACPP_TEST_EMBEDDING_MODEL or allow auto-download.");

        var client = _factory.CreateClient();
        var body = new EmbeddingsRequest
        {
            Input = JsonDocument.Parse("""["cat","kitten","spaceship"]""").RootElement,
        };
        var resp = await client.PostAsJsonAsync("/v1/embeddings", body, TestContext.Current.CancellationToken);
        resp.EnsureSuccessStatusCode();
        var parsed = await resp.Content.ReadFromJsonAsync<EmbeddingsResponse>(
            cancellationToken: TestContext.Current.CancellationToken);

        double simCatKitten    = Cosine(parsed!.Data[0].Embedding, parsed.Data[1].Embedding);
        double simCatSpaceship = Cosine(parsed.Data[0].Embedding,  parsed.Data[2].Embedding);
        Assert.True(simCatKitten > simCatSpaceship,
            $"expected cat/kitten more similar than cat/spaceship; got {simCatKitten:F3} vs {simCatSpaceship:F3}");
    }

    [Fact]
    public async Task Rejects_Unsupported_Encoding_Format()
    {
        if (!_factory.EmbedModelAvailable) Assert.Skip("nomic-embed GGUF unavailable.");

        var client = _factory.CreateClient();
        var body = new EmbeddingsRequest
        {
            Input = JsonDocument.Parse("\"hi\"").RootElement,
            EncodingFormat = "base64",
        };
        var resp = await client.PostAsJsonAsync("/v1/embeddings", body, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Rejects_Null_Input()
    {
        if (!_factory.EmbedModelAvailable) Assert.Skip("nomic-embed GGUF unavailable.");

        var client = _factory.CreateClient();
        // Send "input": null — JsonElement becomes Null kind, handler should 400.
        var raw = """{"input":null}""";
        using var content = new StringContent(raw, System.Text.Encoding.UTF8, "application/json");
        var resp = await client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, resp.StatusCode);
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na  += (double)a[i] * a[i];
            nb  += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-9);
    }

    public sealed class EmbedFactory : WebApplicationFactory<Program>
    {
        public bool EmbedModelAvailable { get; }

        public EmbedFactory()
        {
            var path = TestModelProvider.TryGetEmbeddingModelPath();
            EmbedModelAvailable = !string.IsNullOrWhiteSpace(path) && File.Exists(path);
        }

        protected override IHost CreateHost(IHostBuilder builder)
        {
            var chatPath  = TestModelProvider.EnsureModelPath();
            var embedPath = TestModelProvider.TryGetEmbeddingModelPath();

            builder.ConfigureAppConfiguration(cfg =>
            {
                var settings = new Dictionary<string, string?>
                {
                    ["LlamaServer:ModelPath"]          = chatPath,
                    ["LlamaServer:ContextSize"]        = "512",
                    ["LlamaServer:MaxSequenceCount"]   = "1",
                    ["LlamaServer:GpuLayerCount"]      = "0",
                    ["LlamaServer:OffloadKqv"]         = "false",
                    ["LlamaServer:MaxOutputTokens"]    = "8",
                    ["LlamaServer:Urls"]               = "",
                };
                if (!string.IsNullOrWhiteSpace(embedPath))
                {
                    settings["LlamaServer:EmbeddingModelPath"]     = embedPath;
                    settings["LlamaServer:EmbeddingContextSize"]   = "512";
                    settings["LlamaServer:EmbeddingBatchSize"]     = "512";
                    settings["LlamaServer:EmbeddingGpuLayerCount"] = "0";
                }
                cfg.AddInMemoryCollection(settings);
            });
            return base.CreateHost(builder);
        }
    }
}
