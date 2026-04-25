using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCpp.Bindings.Server.Services;

namespace LlamaCpp.Bindings.Server.Models;

/// <summary>
/// OpenAI-style chat-completions DTOs. V1 covers the subset that matters
/// for a single-model local server: messages + basic sampler knobs +
/// streaming flag. Tool calls, logprobs, function-calling, and multi-
/// content parts are not included — when a caller needs them they're
/// the trigger for a follow-up issue, not something to scaffold blind.
/// </summary>
public sealed class ChatCompletionsRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("messages")]
    public List<ChatMessageDto> Messages { get; set; } = new();

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("seed")]
    public uint? Seed { get; set; }

    [JsonPropertyName("stream")]
    public bool Stream { get; set; }

    /// <summary>
    /// OpenAI-style per-token logit bias map. Keys are token ids (as
    /// strings, since JSON object keys are strings); values are floats
    /// added to the raw logit before sampling. The canonical idiom:
    /// <c>-100</c> effectively bans a token, <c>+100</c> strongly
    /// favours it. Absent or empty = no biasing.
    /// </summary>
    [JsonPropertyName("logit_bias")]
    public Dictionary<string, float>? LogitBias { get; set; }

    // ----- Extended sampler knobs (llama-server parity) -----

    [JsonPropertyName("min_p")]              public float? MinP { get; set; }
    [JsonPropertyName("typical_p")]          public float? TypicalP { get; set; }
    [JsonPropertyName("top_n_sigma")]        public float? TopNSigma { get; set; }

    [JsonPropertyName("xtc_probability")]    public float? XtcProbability { get; set; }
    [JsonPropertyName("xtc_threshold")]      public float? XtcThreshold { get; set; }

    [JsonPropertyName("dry_multiplier")]     public float? DryMultiplier { get; set; }
    [JsonPropertyName("dry_base")]           public float? DryBase { get; set; }
    [JsonPropertyName("dry_allowed_length")] public int? DryAllowedLength { get; set; }
    [JsonPropertyName("dry_penalty_last_n")] public int? DryPenaltyLastN { get; set; }
    [JsonPropertyName("dry_sequence_breakers")] public List<string>? DrySequenceBreakers { get; set; }

    /// <summary>0 = off, 1 = Mirostat v1, 2 = Mirostat v2. When non-zero, overrides truncation + temperature.</summary>
    [JsonPropertyName("mirostat")]           public int? Mirostat { get; set; }
    [JsonPropertyName("mirostat_tau")]       public float? MirostatTau { get; set; }
    [JsonPropertyName("mirostat_eta")]       public float? MirostatEta { get; set; }

    /// <summary>
    /// Dynamic-temperature stretch (llama.cpp's <c>dynatemp_range</c>).
    /// When &gt; 0, the temperature stage flexes around
    /// <see cref="Temperature"/> by ± this amount based on entropy.
    /// </summary>
    [JsonPropertyName("dynatemp_range")]     public float? DynatempRange { get; set; }
    [JsonPropertyName("dynatemp_exponent")]  public float? DynatempExponent { get; set; }

    /// <summary>
    /// Adaptive-p target probability. When &gt;= 0, replaces the standard
    /// greedy / distribution / mirostat terminal with the adaptive-p
    /// sampler. Mutually exclusive with <see cref="Mirostat"/> (mirostat
    /// wins if both are set).
    /// </summary>
    [JsonPropertyName("adaptive_p_target")]  public float? AdaptivePTarget { get; set; }
    [JsonPropertyName("adaptive_p_decay")]   public float? AdaptivePDecay { get; set; }

    /// <summary>
    /// llama-server's <c>samplers</c> field — custom ordering for the
    /// truncation + temperature stages. Allowed names: <c>dry</c>,
    /// <c>top_k</c>, <c>top_p</c>, <c>min_p</c>, <c>typical_p</c>,
    /// <c>top_n_sigma</c>, <c>xtc</c>, <c>temperature</c>. Stages whose
    /// parameters are absent silently skip. Unknown names are HTTP 400.
    /// </summary>
    [JsonPropertyName("samplers")]           public List<string>? Samplers { get; set; }

    [JsonPropertyName("repeat_penalty")]     public float? RepeatPenalty { get; set; }
    [JsonPropertyName("frequency_penalty")]  public float? FrequencyPenalty { get; set; }
    [JsonPropertyName("presence_penalty")]   public float? PresencePenalty { get; set; }
    [JsonPropertyName("repeat_last_n")]      public int? RepeatLastN { get; set; }

    /// <summary>
    /// OpenAI-style stop strings. Accepts either a single string or an
    /// array of up to 4 strings (we enforce the 4-stop cap to match
    /// OpenAI's limit — stops past the fourth are ignored by the official
    /// API and callers sometimes rely on that). Generation halts the
    /// moment the emitted text ends with any stop string; the stop itself
    /// is stripped from the returned content.
    /// </summary>
    [JsonPropertyName("stop")]
    public JsonElement? Stop { get; set; }

    /// <summary>
    /// OpenAI-style <c>response_format</c>. Accepts
    /// <c>{"type":"text"}</c> (no constraint),
    /// <c>{"type":"json_object"}</c> (any valid JSON), or
    /// <c>{"type":"json_schema","json_schema":{"schema":{...}}}</c>
    /// (constrained to the given JSON Schema).
    /// </summary>
    [JsonPropertyName("response_format")]
    public ResponseFormat? ResponseFormat { get; set; }

    /// <summary>
    /// Raw GBNF grammar — llama-server parity. When set, takes precedence
    /// over <see cref="ResponseFormat"/> and <see cref="JsonSchemaShort"/>.
    /// </summary>
    [JsonPropertyName("grammar")]
    public string? Grammar { get; set; }

    /// <summary>
    /// llama-server shortcut: a bare JSON Schema (not wrapped in the
    /// OpenAI <c>response_format</c> envelope). Compiled to GBNF via
    /// <c>JsonSchemaToGbnf</c>. Takes precedence over
    /// <see cref="ResponseFormat"/> but loses to <see cref="Grammar"/>.
    /// </summary>
    [JsonPropertyName("json_schema")]
    public JsonElement? JsonSchemaShort { get; set; }

    // ----- Tool calling (OpenAI-compatible) -----

    /// <summary>
    /// List of tools the model may call. The chat template renders these
    /// into the prompt so tool-capable models know they're available.
    /// </summary>
    [JsonPropertyName("tools")]
    public List<ToolDef>? Tools { get; set; }

    /// <summary>
    /// Polymorphic tool-choice hint. Accepts:
    /// <list type="bullet">
    ///   <item><c>"none"</c> — don't call any tool.</item>
    ///   <item><c>"auto"</c> — let the model decide (default when tools present).</item>
    ///   <item><c>"required"</c> — force any tool call.</item>
    ///   <item><c>{"type":"function","function":{"name":"X"}}</c> — force a specific tool by name.</item>
    /// </list>
    /// </summary>
    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; set; }

    /// <summary>
    /// When <c>true</c>, each choice's response carries a <c>logprobs</c>
    /// object listing per-token log-probabilities. Default <c>false</c>
    /// (no overhead).
    /// </summary>
    [JsonPropertyName("logprobs")]
    public bool? Logprobs { get; set; }

    /// <summary>
    /// When set with <see cref="Logprobs"/> = true, each per-token entry
    /// includes <see cref="LogprobToken.TopLogprobs"/> — the N
    /// highest-logit alternatives at that step. Capped at 20 (OpenAI's
    /// limit) to prevent quadratic cost on long completions.
    /// </summary>
    [JsonPropertyName("top_logprobs")]
    public int? TopLogprobs { get; set; }

    /// <summary>
    /// Opt in to two-model speculative decoding for this request.
    /// Requires the server to have been started with
    /// <c>LlamaServer:DraftModelPath</c> set; ignored otherwise. Falls
    /// back to the standard generator path when the request also uses
    /// images, forced tool calls, or per-token logprobs (none of which
    /// the speculative path supports in V1).
    /// </summary>
    [JsonPropertyName("speculative")]
    public bool? Speculative { get; set; }

    /// <summary>
    /// llama-server's <c>cache_prompt</c>. Default <c>true</c> — the
    /// SessionPool reuses the longest common prefix it can find. Set
    /// <c>false</c> to force a cold decode (useful for determinism
    /// testing and one-shot requests that shouldn't warm the cache for
    /// later callers).
    /// </summary>
    [JsonPropertyName("cache_prompt")]
    public bool? CachePrompt { get; set; }
}

public sealed class ChatMessageDto
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    /// <summary>
    /// Polymorphic content: a plain string (the legacy and still-most-common
    /// shape) or an array of <see cref="ContentPart"/> entries for
    /// multimodal chat (text interleaved with images). May be null when
    /// the message is an assistant tool-call turn (<see cref="ToolCalls"/>
    /// is populated instead).
    /// </summary>
    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public MessageContent? Content { get; set; }

    /// <summary>
    /// For <c>role: "tool"</c> messages — matches the <c>id</c> of the
    /// assistant-side tool_call this message is responding to.
    /// </summary>
    [JsonPropertyName("tool_call_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ToolCallId { get; set; }

    /// <summary>
    /// Assistant-side tool calls. Set on the response when the model has
    /// been grammar-forced into a specific tool invocation. Also accepted
    /// on incoming history messages (role=assistant) so multi-turn tool
    /// flows can round-trip.
    /// </summary>
    [JsonPropertyName("tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<ToolCall>? ToolCalls { get; set; }
}

/// <summary>
/// Request-side wrapper over OpenAI's polymorphic chat-content field.
/// Exactly one of <see cref="Text"/> or <see cref="Parts"/> is set after
/// deserialization. Implicit conversion from <see cref="string"/> keeps
/// caller sites that write <c>Content = "hi"</c> working.
/// </summary>
[JsonConverter(typeof(MessageContentConverter))]
public sealed class MessageContent
{
    /// <summary>Set when the wire content is a bare string.</summary>
    public string? Text { get; set; }

    /// <summary>Set when the wire content is an array of parts.</summary>
    public List<ContentPart>? Parts { get; set; }

    public static implicit operator MessageContent?(string? text) =>
        text is null ? null : new() { Text = text };
}

/// <summary>
/// One entry in a multimodal content array: either a text run or an
/// image reference. OpenAI's shape: <c>{type: "text", text: "..."}</c>
/// or <c>{type: "image_url", image_url: {url: "..."}}</c>. Audio /
/// video parts aren't modelled here.
/// </summary>
public sealed class ContentPart
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "text";

    [JsonPropertyName("text")]
    public string? Text { get; set; }

    [JsonPropertyName("image_url")]
    public ImageUrl? ImageUrl { get; set; }
}

public sealed class ImageUrl
{
    [JsonPropertyName("url")]
    public string Url { get; set; } = "";
}

// ----- Tool-calling DTOs -----

/// <summary>
/// OpenAI tool definition: <c>{"type":"function","function":{...}}</c>.
/// The outer envelope leaves room for future non-function tools
/// (OpenAI has been teasing retrieval + code interpreter); for now
/// only <c>type = "function"</c> is meaningful.
/// </summary>
public sealed class ToolDef
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    [JsonPropertyName("function")]
    public ToolFunctionDef? Function { get; set; }
}

public sealed class ToolFunctionDef
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    /// <summary>
    /// JSON Schema describing the function's argument object. Compiled
    /// to GBNF via <see cref="JsonSchemaToGbnf"/> when the server
    /// forces a tool call.
    /// </summary>
    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; set; }
}

/// <summary>
/// One tool call emitted by the assistant.
/// </summary>
public sealed class ToolCall
{
    /// <summary>Unique id for the call — clients match <c>role=tool</c> responses to it.</summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    [JsonPropertyName("function")]
    public ToolCallFunction Function { get; set; } = new();
}

public sealed class ToolCallFunction
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    /// <summary>
    /// The function arguments as a JSON string (OpenAI's convention —
    /// the arguments object is returned stringified so clients can
    /// parse once they know which function was called).
    /// </summary>
    [JsonPropertyName("arguments")]
    public string Arguments { get; set; } = "{}";
}

internal sealed class MessageContentConverter : JsonConverter<MessageContent>
{
    public override MessageContent? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        switch (reader.TokenType)
        {
            case JsonTokenType.Null:
                return null;
            case JsonTokenType.String:
                return new MessageContent { Text = reader.GetString() };
            case JsonTokenType.StartArray:
                var parts = JsonSerializer.Deserialize<List<ContentPart>>(ref reader, options);
                return new MessageContent { Parts = parts };
            default:
                throw new JsonException(
                    $"chat message content must be a string or an array of content parts (got {reader.TokenType}).");
        }
    }

    public override void Write(Utf8JsonWriter writer, MessageContent value, JsonSerializerOptions options)
    {
        if (value.Text is not null)
        {
            writer.WriteStringValue(value.Text);
        }
        else if (value.Parts is not null)
        {
            JsonSerializer.Serialize(writer, value.Parts, options);
        }
        else
        {
            writer.WriteNullValue();
        }
    }
}

public sealed class ChatCompletionsResponse
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("choices")]
    public List<ChatChoice> Choices { get; set; } = new();

    /// <summary>
    /// llama-server-compatible per-request timing breakdown. Included on
    /// non-streaming responses and in the final SSE chunk on streaming.
    /// </summary>
    [JsonPropertyName("timings")]
    public RequestTimings? Timings { get; set; }
}

/// <summary>
/// Per-request timing + token-count breakdown. <c>prompt_ms</c> is the
/// wall-clock spent ingesting the new prompt tokens (skipping anything
/// the pool's prefix cache satisfied); <c>predicted_ms</c> is the time
/// from the first emitted token to the last. Token-per-ms fields are
/// derived; they're there so clients don't have to compute them.
/// </summary>
public sealed class RequestTimings
{
    [JsonPropertyName("prompt_n")]            public int PromptN { get; set; }
    [JsonPropertyName("prompt_ms")]           public double PromptMs { get; set; }
    [JsonPropertyName("prompt_per_token_ms")] public double PromptPerTokenMs { get; set; }
    [JsonPropertyName("predicted_n")]         public int PredictedN { get; set; }
    [JsonPropertyName("predicted_ms")]        public double PredictedMs { get; set; }
    [JsonPropertyName("predicted_per_token_ms")] public double PredictedPerTokenMs { get; set; }
    [JsonPropertyName("cached_n")]            public int CachedN { get; set; }
}

public sealed class ChatChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("message")]
    public ChatMessageDto Message { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; set; } = "stop";

    /// <summary>Per-token log-probabilities, populated when the request set <c>logprobs: true</c>.</summary>
    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsContent? Logprobs { get; set; }
}

/// <summary>
/// OpenAI's <c>choices[].logprobs</c> envelope. <see cref="Content"/>
/// holds one entry per generated token, in emission order.
/// </summary>
public sealed class LogprobsContent
{
    [JsonPropertyName("content")]
    public List<LogprobToken> Content { get; set; } = new();
}

public sealed class LogprobToken
{
    /// <summary>The token's text rendering (the same string the chunk's <c>delta.content</c> would have carried).</summary>
    [JsonPropertyName("token")]
    public string Token { get; set; } = "";

    /// <summary>Natural log of the token's probability under the full vocabulary distribution.</summary>
    [JsonPropertyName("logprob")]
    public float Logprob { get; set; }

    /// <summary>The token's UTF-8 byte sequence, or <c>null</c> when the token isn't representable as UTF-8.</summary>
    [JsonPropertyName("bytes")]
    public int[]? Bytes { get; set; }

    /// <summary>Top-N alternatives at this position, populated when the request set <c>top_logprobs</c>.</summary>
    [JsonPropertyName("top_logprobs")]
    public List<TopLogprob> TopLogprobs { get; set; } = new();
}

public sealed class TopLogprob
{
    [JsonPropertyName("token")]
    public string Token { get; set; } = "";

    [JsonPropertyName("logprob")]
    public float Logprob { get; set; }

    [JsonPropertyName("bytes")]
    public int[]? Bytes { get; set; }
}

/// <summary>SSE chunk emitted while <c>stream=true</c>.</summary>
public sealed class ChatCompletionsChunk
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public long Created { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("choices")]
    public List<ChatChunkChoice> Choices { get; set; } = new();

    /// <summary>Set only on the final chunk when the caller wants timings.</summary>
    [JsonPropertyName("timings")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public RequestTimings? Timings { get; set; }
}

public sealed class ChatChunkChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("delta")]
    public ChatDelta Delta { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }

    /// <summary>Per-chunk logprobs (one entry per token in this delta).</summary>
    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsContent? Logprobs { get; set; }
}

public sealed class ChatDelta
{
    [JsonPropertyName("role")]
    public string? Role { get; set; }

    [JsonPropertyName("content")]
    public string? Content { get; set; }
}

/// <summary>
/// llama-server's native <c>/completion</c> body. Minimal — prompt in,
/// text out. Streaming follows the same SSE shape as chat completions.
/// </summary>
public sealed class CompletionRequest
{
    [JsonPropertyName("prompt")]
    public string Prompt { get; set; } = "";

    [JsonPropertyName("n_predict")]
    public int? MaxTokens { get; set; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("seed")]
    public uint? Seed { get; set; }

    [JsonPropertyName("stream")]
    public bool Stream { get; set; }

    /// <summary>Per-token logit bias, same semantics as <see cref="ChatCompletionsRequest.LogitBias"/>.</summary>
    [JsonPropertyName("logit_bias")]
    public Dictionary<string, float>? LogitBias { get; set; }

    // ----- Extended sampler knobs (llama-server parity) -----

    [JsonPropertyName("min_p")]              public float? MinP { get; set; }
    [JsonPropertyName("typical_p")]          public float? TypicalP { get; set; }
    [JsonPropertyName("top_n_sigma")]        public float? TopNSigma { get; set; }

    [JsonPropertyName("xtc_probability")]    public float? XtcProbability { get; set; }
    [JsonPropertyName("xtc_threshold")]      public float? XtcThreshold { get; set; }

    [JsonPropertyName("dry_multiplier")]     public float? DryMultiplier { get; set; }
    [JsonPropertyName("dry_base")]           public float? DryBase { get; set; }
    [JsonPropertyName("dry_allowed_length")] public int? DryAllowedLength { get; set; }
    [JsonPropertyName("dry_penalty_last_n")] public int? DryPenaltyLastN { get; set; }
    [JsonPropertyName("dry_sequence_breakers")] public List<string>? DrySequenceBreakers { get; set; }

    [JsonPropertyName("mirostat")]           public int? Mirostat { get; set; }
    [JsonPropertyName("mirostat_tau")]       public float? MirostatTau { get; set; }
    [JsonPropertyName("mirostat_eta")]       public float? MirostatEta { get; set; }

    [JsonPropertyName("dynatemp_range")]     public float? DynatempRange { get; set; }
    [JsonPropertyName("dynatemp_exponent")]  public float? DynatempExponent { get; set; }

    [JsonPropertyName("adaptive_p_target")]  public float? AdaptivePTarget { get; set; }
    [JsonPropertyName("adaptive_p_decay")]   public float? AdaptivePDecay { get; set; }

    [JsonPropertyName("samplers")]           public List<string>? Samplers { get; set; }

    [JsonPropertyName("repeat_penalty")]     public float? RepeatPenalty { get; set; }
    [JsonPropertyName("frequency_penalty")]  public float? FrequencyPenalty { get; set; }
    [JsonPropertyName("presence_penalty")]   public float? PresencePenalty { get; set; }
    [JsonPropertyName("repeat_last_n")]      public int? RepeatLastN { get; set; }

    /// <summary>Stop strings, same semantics as <see cref="ChatCompletionsRequest.Stop"/>.</summary>
    [JsonPropertyName("stop")]
    public JsonElement? Stop { get; set; }

    [JsonPropertyName("response_format")]
    public ResponseFormat? ResponseFormat { get; set; }

    [JsonPropertyName("grammar")]
    public string? Grammar { get; set; }

    [JsonPropertyName("json_schema")]
    public JsonElement? JsonSchemaShort { get; set; }

    /// <summary>llama-server's <c>cache_prompt</c>. Default true; same semantics as the chat endpoint.</summary>
    [JsonPropertyName("cache_prompt")]
    public bool? CachePrompt { get; set; }
}

public sealed class CompletionResponse
{
    [JsonPropertyName("content")]
    public string Content { get; set; } = "";

    [JsonPropertyName("stop_reason")]
    public string StopReason { get; set; } = "stop";

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("timings")]
    public RequestTimings? Timings { get; set; }
}

public sealed class ModelsListResponse
{
    [JsonPropertyName("object")]
    public string Object { get; set; } = "list";

    [JsonPropertyName("data")]
    public List<ModelEntry> Data { get; set; } = new();
}

public sealed class ModelEntry
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "model";

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; set; } = "local";
}

/// <summary>
/// OpenAI's <c>response_format</c> object. <see cref="Type"/> drives
/// everything:
/// <list type="bullet">
///   <item><c>"text"</c> — no constraint (default).</item>
///   <item><c>"json_object"</c> — any valid JSON.</item>
///   <item><c>"json_schema"</c> — requires <see cref="JsonSchema"/>.</item>
/// </list>
/// </summary>
public sealed class ResponseFormat
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "text";

    [JsonPropertyName("json_schema")]
    public JsonSchemaSpec? JsonSchema { get; set; }
}

/// <summary>
/// OpenAI's <c>response_format.json_schema</c> envelope. The actual
/// schema lives in <see cref="Schema"/>; <see cref="Name"/> is
/// cosmetic (surfaced in error messages), <see cref="Strict"/> is an
/// OpenAI flag we honour implicitly (GBNF is always strict).
/// </summary>
public sealed class JsonSchemaSpec
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("schema")]
    public JsonElement? Schema { get; set; }

    [JsonPropertyName("strict")]
    public bool? Strict { get; set; }
}

/// <summary>
/// Normalise the polymorphic <c>stop</c> field into a <c>string[]</c>.
/// Shared between chat and completion endpoints.
/// </summary>
internal static class StopNormalizer
{
    /// <summary>Max number of stop strings honoured per request (OpenAI parity).</summary>
    public const int MaxStops = 4;

    /// <summary>
    /// Returns the parsed stop array, or null/empty when the field is
    /// absent. Throws <see cref="ArgumentException"/> on malformed input
    /// (non-string array entries, object/number/boolean values).
    /// </summary>
    public static string[]? Parse(JsonElement? element)
    {
        if (element is not JsonElement el) return null;
        switch (el.ValueKind)
        {
            case JsonValueKind.Undefined:
            case JsonValueKind.Null:
                return null;
            case JsonValueKind.String:
            {
                var s = el.GetString();
                return string.IsNullOrEmpty(s) ? null : new[] { s };
            }
            case JsonValueKind.Array:
            {
                var list = new List<string>();
                foreach (var item in el.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.String)
                    {
                        throw new ArgumentException(
                            $"stop array entries must be strings (got {item.ValueKind}).");
                    }
                    var s = item.GetString();
                    if (!string.IsNullOrEmpty(s)) list.Add(s);
                    if (list.Count >= MaxStops) break;
                }
                return list.Count == 0 ? null : list.ToArray();
            }
            default:
                throw new ArgumentException(
                    $"stop must be a string or array of strings (got {el.ValueKind}).");
        }
    }
}

/// <summary>
/// Projections from request DTOs to the shared <see cref="SamplerParams"/>.
/// Keeps the field-by-field copy out of the endpoint handlers.
/// </summary>
internal static class SamplerParamsExtensions
{
    public static SamplerParams ToSamplerParams(this ChatCompletionsRequest r) => new()
    {
        Temperature         = r.Temperature,
        Seed                = r.Seed,
        DynatempRange       = r.DynatempRange,
        DynatempExponent    = r.DynatempExponent,
        Mirostat            = r.Mirostat,
        MirostatTau         = r.MirostatTau,
        MirostatEta         = r.MirostatEta,
        AdaptivePTarget     = r.AdaptivePTarget,
        AdaptivePDecay      = r.AdaptivePDecay,
        TopK                = r.TopK,
        TopP                = r.TopP,
        MinP                = r.MinP,
        TypicalP            = r.TypicalP,
        TopNSigma           = r.TopNSigma,
        XtcProbability      = r.XtcProbability,
        XtcThreshold        = r.XtcThreshold,
        DryMultiplier       = r.DryMultiplier,
        DryBase             = r.DryBase,
        DryAllowedLength    = r.DryAllowedLength,
        DryPenaltyLastN     = r.DryPenaltyLastN,
        DrySequenceBreakers = r.DrySequenceBreakers,
        RepeatPenalty       = r.RepeatPenalty,
        FrequencyPenalty    = r.FrequencyPenalty,
        PresencePenalty     = r.PresencePenalty,
        RepeatLastN         = r.RepeatLastN,
        LogitBias           = r.LogitBias,
        Samplers            = r.Samplers,
        // Grammar is resolved separately by the endpoint and assigned to
        // SamplerParams after this projection — keeps JSON-parse errors
        // (malformed schema, unknown response_format.type) out of the
        // DTO layer.
    };

    public static SamplerParams ToSamplerParams(this CompletionRequest r) => new()
    {
        Temperature         = r.Temperature,
        Seed                = r.Seed,
        DynatempRange       = r.DynatempRange,
        DynatempExponent    = r.DynatempExponent,
        Mirostat            = r.Mirostat,
        MirostatTau         = r.MirostatTau,
        MirostatEta         = r.MirostatEta,
        AdaptivePTarget     = r.AdaptivePTarget,
        AdaptivePDecay      = r.AdaptivePDecay,
        TopK                = r.TopK,
        TopP                = r.TopP,
        MinP                = r.MinP,
        TypicalP            = r.TypicalP,
        TopNSigma           = r.TopNSigma,
        XtcProbability      = r.XtcProbability,
        XtcThreshold        = r.XtcThreshold,
        DryMultiplier       = r.DryMultiplier,
        DryBase             = r.DryBase,
        DryAllowedLength    = r.DryAllowedLength,
        DryPenaltyLastN     = r.DryPenaltyLastN,
        DrySequenceBreakers = r.DrySequenceBreakers,
        RepeatPenalty       = r.RepeatPenalty,
        FrequencyPenalty    = r.FrequencyPenalty,
        PresencePenalty     = r.PresencePenalty,
        RepeatLastN         = r.RepeatLastN,
        LogitBias           = r.LogitBias,
        Samplers            = r.Samplers,
        // Grammar is resolved separately by the endpoint and assigned to
        // SamplerParams after this projection — keeps JSON-parse errors
        // (malformed schema, unknown response_format.type) out of the
        // DTO layer.
    };
}
