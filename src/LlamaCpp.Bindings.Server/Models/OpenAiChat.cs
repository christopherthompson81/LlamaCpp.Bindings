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

    [JsonPropertyName("repeat_penalty")]     public float? RepeatPenalty { get; set; }
    [JsonPropertyName("frequency_penalty")]  public float? FrequencyPenalty { get; set; }
    [JsonPropertyName("presence_penalty")]   public float? PresencePenalty { get; set; }
    [JsonPropertyName("repeat_last_n")]      public int? RepeatLastN { get; set; }

    /// <summary>Stop strings, same semantics as <see cref="ChatCompletionsRequest.Stop"/>.</summary>
    [JsonPropertyName("stop")]
    public JsonElement? Stop { get; set; }
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
        Mirostat            = r.Mirostat,
        MirostatTau         = r.MirostatTau,
        MirostatEta         = r.MirostatEta,
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
    };

    public static SamplerParams ToSamplerParams(this CompletionRequest r) => new()
    {
        Temperature         = r.Temperature,
        Seed                = r.Seed,
        Mirostat            = r.Mirostat,
        MirostatTau         = r.MirostatTau,
        MirostatEta         = r.MirostatEta,
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
    };
}
