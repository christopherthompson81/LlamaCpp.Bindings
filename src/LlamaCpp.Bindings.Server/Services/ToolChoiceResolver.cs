using System.Text.Json;
using LlamaCpp.Bindings.Server.Models;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Parses the polymorphic OpenAI <c>tool_choice</c> field into a
/// strongly-typed descriptor and decides whether a grammar should be
/// forced for the sampler. V1 only forces when a specific function is
/// named — the <c>"required"</c> (any-tool) variant would need a GBNF
/// union of every tool's schema, which <see cref="JsonSchemaToGbnf"/>
/// doesn't support out-of-the-box. That scope is deferred.
/// </summary>
public enum ToolChoiceKind
{
    /// <summary>No tools / no constraint. Either the caller omitted the field or set it to <c>"auto"</c>.</summary>
    Auto,
    /// <summary>Caller explicitly set <c>"none"</c>: tools are NOT rendered into the prompt.</summary>
    None,
    /// <summary>Caller set <c>"required"</c>: any tool call must be emitted. V1 rejects this with 400.</summary>
    RequiredAny,
    /// <summary>Caller specified a single tool by name. Forces a grammar for that tool's parameters schema.</summary>
    ForcedFunction,
}

public sealed class ToolChoiceDescriptor
{
    public ToolChoiceKind Kind { get; init; }
    /// <summary>When <see cref="Kind"/> is <see cref="ToolChoiceKind.ForcedFunction"/>.</summary>
    public ToolDef? ForcedTool { get; init; }
}

public static class ToolChoiceResolver
{
    /// <summary>
    /// Parse <paramref name="toolChoice"/> (may be null, a string, or an
    /// object) into a <see cref="ToolChoiceDescriptor"/>. Throws
    /// <see cref="ArgumentException"/> for malformed input; callers
    /// surface HTTP 400.
    /// </summary>
    public static ToolChoiceDescriptor Resolve(
        JsonElement? toolChoice, IReadOnlyList<ToolDef>? tools)
    {
        bool hasTools = tools is { Count: > 0 };

        // No tool_choice specified → default is "auto" when tools are
        // present, otherwise "none" (we don't include tools in the
        // prompt at all).
        if (toolChoice is not JsonElement el ||
            el.ValueKind == JsonValueKind.Null ||
            el.ValueKind == JsonValueKind.Undefined)
        {
            return new() { Kind = hasTools ? ToolChoiceKind.Auto : ToolChoiceKind.None };
        }

        if (el.ValueKind == JsonValueKind.String)
        {
            var s = el.GetString() ?? "";
            return s switch
            {
                "auto" => new ToolChoiceDescriptor { Kind = ToolChoiceKind.Auto },
                "none" => new ToolChoiceDescriptor { Kind = ToolChoiceKind.None },
                "required" => new ToolChoiceDescriptor { Kind = ToolChoiceKind.RequiredAny },
                _ => throw new ArgumentException(
                    $"tool_choice='{s}' is not recognised. Use 'auto', 'none', 'required', or " +
                    "an object of the form {\"type\":\"function\",\"function\":{\"name\":\"X\"}}."),
            };
        }

        if (el.ValueKind == JsonValueKind.Object)
        {
            if (!el.TryGetProperty("type", out var typeEl) ||
                typeEl.ValueKind != JsonValueKind.String ||
                typeEl.GetString() != "function")
            {
                throw new ArgumentException(
                    "tool_choice object must be {\"type\":\"function\",\"function\":{\"name\":\"X\"}}.");
            }
            if (!el.TryGetProperty("function", out var fnEl) ||
                fnEl.ValueKind != JsonValueKind.Object ||
                !fnEl.TryGetProperty("name", out var nameEl) ||
                nameEl.ValueKind != JsonValueKind.String)
            {
                throw new ArgumentException(
                    "tool_choice.function.name is required when tool_choice.type is 'function'.");
            }
            var name = nameEl.GetString()!;
            var matched = tools?.FirstOrDefault(t =>
                t.Function is { } f && f.Name == name);
            if (matched is null)
            {
                throw new ArgumentException(
                    $"tool_choice references function '{name}', but no tool with that name was provided in tools[].");
            }
            return new ToolChoiceDescriptor
            {
                Kind = ToolChoiceKind.ForcedFunction,
                ForcedTool = matched,
            };
        }

        throw new ArgumentException(
            $"tool_choice must be a string or object (got {el.ValueKind}).");
    }

    /// <summary>
    /// Build a <see cref="LlamaGrammar"/> that constrains output to the
    /// named tool's JSON Schema. Returns null when <paramref name="descriptor"/>
    /// isn't <see cref="ToolChoiceKind.ForcedFunction"/>.
    /// </summary>
    public static LlamaGrammar? ForcedGrammar(ToolChoiceDescriptor descriptor)
    {
        if (descriptor.Kind != ToolChoiceKind.ForcedFunction) return null;
        var tool = descriptor.ForcedTool!;
        var fn = tool.Function!;
        if (fn.Parameters is not JsonElement schema ||
            schema.ValueKind == JsonValueKind.Null ||
            schema.ValueKind == JsonValueKind.Undefined)
        {
            // No schema — constrain to any JSON object so we still get a
            // parseable tool call out.
            return LlamaGrammar.Json;
        }
        try
        {
            var gbnf = JsonSchemaToGbnf.Convert(schema);
            return new LlamaGrammar(gbnf, StartRuleName: "root");
        }
        catch (Exception ex) when (ex is not ArgumentException)
        {
            throw new ArgumentException(
                $"Failed to compile tool '{fn.Name}' parameters schema to GBNF: " + ex.Message, ex);
        }
    }
}
