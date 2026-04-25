using System.Text.Json;
using LlamaCpp.Bindings.Server.Models;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Resolves the "what grammar should we constrain output with, if any"
/// question from a request's <c>response_format</c> / <c>grammar</c> /
/// <c>json_schema</c> fields. Shared between chat and completion
/// endpoints so they can't drift on precedence rules.
/// </summary>
/// <remarks>
/// <para><b>Precedence:</b> <c>grammar</c> &gt; <c>json_schema</c> &gt;
/// <c>response_format</c>. Explicit GBNF wins because it's the most
/// specific — the caller has already made the trade-off between
/// expressiveness and flexibility. <c>json_schema</c> beats
/// <c>response_format</c> because llama-server's shortcut form sits
/// closer to the sampler semantically (it's a bare schema, no
/// envelope).</para>
///
/// <para><b>Error surface:</b> every method on this class throws
/// <see cref="ArgumentException"/> for malformed input (missing schema
/// inside <c>json_schema</c> mode, unknown <c>response_format.type</c>,
/// empty <c>grammar</c> string). Callers surface these as HTTP 400.</para>
/// </remarks>
public static class GrammarFactory
{
    /// <summary>
    /// Returns the resolved <see cref="LlamaGrammar"/> to attach to the
    /// sampler chain, or <c>null</c> when no grammar constraint is
    /// requested.
    /// </summary>
    public static LlamaGrammar? Resolve(
        string? rawGrammar,
        JsonElement? jsonSchemaShort,
        ResponseFormat? responseFormat)
    {
        // 1. Explicit GBNF wins.
        if (!string.IsNullOrWhiteSpace(rawGrammar))
        {
            return new LlamaGrammar(rawGrammar!, StartRuleName: "root");
        }

        // 2. llama-server-style bare `json_schema` shortcut.
        if (jsonSchemaShort is JsonElement shortSchema && shortSchema.ValueKind != JsonValueKind.Null && shortSchema.ValueKind != JsonValueKind.Undefined)
        {
            return CompileSchema(shortSchema);
        }

        // 3. OpenAI-style response_format envelope.
        if (responseFormat is not null)
        {
            return ResolveResponseFormat(responseFormat);
        }

        return null;
    }

    private static LlamaGrammar? ResolveResponseFormat(ResponseFormat rf)
    {
        var type = rf.Type ?? "text";
        switch (type)
        {
            case "text":
                return null;
            case "json_object":
                return LlamaGrammar.Json;
            case "json_schema":
                if (rf.JsonSchema?.Schema is not JsonElement schema ||
                    schema.ValueKind == JsonValueKind.Null ||
                    schema.ValueKind == JsonValueKind.Undefined)
                {
                    throw new ArgumentException(
                        "response_format.type='json_schema' requires response_format.json_schema.schema " +
                        "(the actual JSON Schema object).");
                }
                return CompileSchema(schema);
            default:
                throw new ArgumentException(
                    $"response_format.type='{type}' is not supported. Use 'text', 'json_object', or 'json_schema'.");
        }
    }

    private static LlamaGrammar CompileSchema(JsonElement schema)
    {
        // JsonSchemaToGbnf throws for malformed schemas; rethrow as
        // ArgumentException so the endpoint can surface HTTP 400.
        try
        {
            var gbnf = JsonSchemaToGbnf.Convert(schema);
            return new LlamaGrammar(gbnf, StartRuleName: "root");
        }
        catch (Exception ex) when (ex is not ArgumentException)
        {
            throw new ArgumentException(
                "Failed to compile JSON Schema to GBNF: " + ex.Message, ex);
        }
    }
}
