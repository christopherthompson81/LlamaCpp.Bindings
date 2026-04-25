using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LlamaCpp.Bindings.LlamaChat.Services.Remote;

/// <summary>
/// Polymorphic content serializer matching OpenAI's wire shape: either a bare
/// string (legacy) or an array of <see cref="ContentPart"/> entries (multimodal).
/// Mirrors the converter in <c>LlamaCpp.Bindings.Server.Models.OpenAiChat</c>.
/// </summary>
internal sealed class MessageContentConverter : JsonConverter<MessageContent>
{
    public override MessageContent? Read(ref Utf8JsonReader reader, System.Type typeToConvert, JsonSerializerOptions options)
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
