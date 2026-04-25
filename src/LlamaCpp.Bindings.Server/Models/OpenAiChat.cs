using System.Text.Json.Serialization;

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
}

public sealed class ChatMessageDto
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    [JsonPropertyName("content")]
    public string Content { get; set; } = "";
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
}

public sealed class ChatChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("message")]
    public ChatMessageDto Message { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; set; } = "stop";
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
}

public sealed class ChatChunkChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("delta")]
    public ChatDelta Delta { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }
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
}

public sealed class CompletionResponse
{
    [JsonPropertyName("content")]
    public string Content { get; set; } = "";

    [JsonPropertyName("stop_reason")]
    public string StopReason { get; set; } = "stop";

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";
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
