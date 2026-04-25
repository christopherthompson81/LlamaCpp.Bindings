using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings.LlamaChat.Services.Remote;

/// <summary>
/// Client-side DTOs for OpenAI-compatible <c>/v1/chat/completions</c> and
/// <c>/v1/models</c>. A thin slice of the server's
/// <c>LlamaCpp.Bindings.Server.Models.OpenAiChat</c> — only the fields LlamaChat
/// actually populates or reads. All request fields are nullable so we only
/// emit what the user has set; <c>JsonIgnoreCondition.WhenWritingNull</c> on
/// the serializer keeps the wire body small.
/// </summary>
public sealed class ChatCompletionsRequest
{
    [JsonPropertyName("model")]            public string? Model { get; set; }
    [JsonPropertyName("messages")]         public List<OpenAiChatMessage> Messages { get; set; } = new();
    [JsonPropertyName("max_tokens")]       public int? MaxTokens { get; set; }
    [JsonPropertyName("temperature")]      public float? Temperature { get; set; }
    [JsonPropertyName("top_p")]            public float? TopP { get; set; }
    [JsonPropertyName("top_k")]            public int? TopK { get; set; }
    [JsonPropertyName("seed")]             public uint? Seed { get; set; }
    [JsonPropertyName("stream")]           public bool Stream { get; set; }

    [JsonPropertyName("min_p")]            public float? MinP { get; set; }
    [JsonPropertyName("typical_p")]        public float? TypicalP { get; set; }
    [JsonPropertyName("top_n_sigma")]      public float? TopNSigma { get; set; }

    [JsonPropertyName("xtc_probability")]  public float? XtcProbability { get; set; }
    [JsonPropertyName("xtc_threshold")]    public float? XtcThreshold { get; set; }

    [JsonPropertyName("dry_multiplier")]     public float? DryMultiplier { get; set; }
    [JsonPropertyName("dry_base")]           public float? DryBase { get; set; }
    [JsonPropertyName("dry_allowed_length")] public int? DryAllowedLength { get; set; }
    [JsonPropertyName("dry_penalty_last_n")] public int? DryPenaltyLastN { get; set; }

    [JsonPropertyName("mirostat")]         public int? Mirostat { get; set; }
    [JsonPropertyName("mirostat_tau")]     public float? MirostatTau { get; set; }
    [JsonPropertyName("mirostat_eta")]     public float? MirostatEta { get; set; }

    [JsonPropertyName("dynatemp_range")]    public float? DynatempRange { get; set; }
    [JsonPropertyName("dynatemp_exponent")] public float? DynatempExponent { get; set; }

    [JsonPropertyName("repeat_penalty")]   public float? RepeatPenalty { get; set; }
    [JsonPropertyName("frequency_penalty")] public float? FrequencyPenalty { get; set; }
    [JsonPropertyName("presence_penalty")] public float? PresencePenalty { get; set; }
    [JsonPropertyName("repeat_last_n")]    public int? RepeatLastN { get; set; }

    [JsonPropertyName("cache_prompt")]     public bool? CachePrompt { get; set; }

    /// <summary>
    /// Free-form kwargs forwarded into the server's chat-template Jinja
    /// context. Mirrors llama.cpp server's <c>chat_template_kwargs</c>;
    /// common usage is <c>{"enable_thinking": false}</c> to suppress
    /// reasoning preambles for short utility calls (title gen).
    /// </summary>
    [JsonPropertyName("chat_template_kwargs")]
    public Dictionary<string, object?>? ChatTemplateKwargs { get; set; }
}

public sealed class OpenAiChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    /// <summary>
    /// String content (the typical case) or a content-part array (multimodal).
    /// Both shapes round-trip through <see cref="MessageContent"/>'s converter.
    /// </summary>
    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public MessageContent? Content { get; set; }
}

[JsonConverter(typeof(MessageContentConverter))]
public sealed class MessageContent
{
    public string? Text { get; set; }
    public List<ContentPart>? Parts { get; set; }

    public static MessageContent FromText(string text) => new() { Text = text };
    public static MessageContent FromParts(List<ContentPart> parts) => new() { Parts = parts };
}

public sealed class ContentPart
{
    [JsonPropertyName("type")]      public string Type { get; set; } = "text";
    [JsonPropertyName("text")]      public string? Text { get; set; }
    [JsonPropertyName("image_url")] public ImageUrl? ImageUrl { get; set; }
}

public sealed class ImageUrl
{
    [JsonPropertyName("url")] public string Url { get; set; } = "";
}

public sealed class ChatCompletionsChunk
{
    [JsonPropertyName("id")]      public string Id { get; set; } = "";
    [JsonPropertyName("model")]   public string Model { get; set; } = "";
    [JsonPropertyName("choices")] public List<ChatChunkChoice> Choices { get; set; } = new();
    [JsonPropertyName("timings")] public RequestTimings? Timings { get; set; }
}

public sealed class ChatChunkChoice
{
    [JsonPropertyName("index")]         public int Index { get; set; }
    [JsonPropertyName("delta")]         public ChatDelta Delta { get; set; } = new();
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
}

public sealed class ChatDelta
{
    [JsonPropertyName("role")]    public string? Role { get; set; }
    [JsonPropertyName("content")] public string? Content { get; set; }
}

public sealed class ChatCompletionsResponse
{
    [JsonPropertyName("id")]      public string Id { get; set; } = "";
    [JsonPropertyName("model")]   public string Model { get; set; } = "";
    [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = new();
    [JsonPropertyName("timings")] public RequestTimings? Timings { get; set; }
}

public sealed class ChatChoice
{
    [JsonPropertyName("index")]         public int Index { get; set; }
    [JsonPropertyName("message")]       public OpenAiChatMessage Message { get; set; } = new();
    [JsonPropertyName("finish_reason")] public string FinishReason { get; set; } = "stop";
}

public sealed class RequestTimings
{
    [JsonPropertyName("prompt_n")]     public int PromptN { get; set; }
    [JsonPropertyName("prompt_ms")]    public double PromptMs { get; set; }
    [JsonPropertyName("predicted_n")]  public int PredictedN { get; set; }
    [JsonPropertyName("predicted_ms")] public double PredictedMs { get; set; }
    [JsonPropertyName("cached_n")]     public int CachedN { get; set; }
}

public sealed class ModelsListResponse
{
    [JsonPropertyName("data")] public List<ModelEntry> Data { get; set; } = new();
}

public sealed class ModelEntry
{
    [JsonPropertyName("id")]       public string Id { get; set; } = "";
    [JsonPropertyName("owned_by")] public string OwnedBy { get; set; } = "";
}
