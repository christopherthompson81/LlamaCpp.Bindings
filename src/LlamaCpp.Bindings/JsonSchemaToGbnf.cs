using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings;

/// <summary>
/// Converts a JSON Schema into a GBNF grammar string compatible with
/// llama.cpp's grammar-based sampler. Behavioural port of
/// <c>llama.cpp/common/json-schema-to-grammar.cpp</c> — covers the same
/// surface that <c>llama-server</c>'s response-format endpoint exposes:
/// primitive types, arrays (<c>items</c>/<c>prefixItems</c>, <c>minItems</c>
/// /<c>maxItems</c>), objects (<c>properties</c>, <c>required</c>,
/// <c>additionalProperties</c>), combinators (<c>oneOf</c>/<c>anyOf</c>/
/// <c>allOf</c>), local <c>$ref</c> chains, <c>enum</c>, <c>const</c>,
/// string formats (date / time / date-time / uuid), string length
/// constraints, integer ranges, and regex <c>pattern</c> compilation.
/// </summary>
/// <remarks>
/// The output is the plain-text grammar; feed it to
/// <see cref="LlamaGrammar"/> alongside a start rule (<c>root</c> by
/// default). Remote <c>$ref</c> fetching is intentionally unsupported —
/// local (<c>#/</c>) refs work; <c>https://</c> refs will throw.
/// Errors accumulated during conversion (unknown types, malformed
/// patterns, missing refs) throw a <see cref="JsonSchemaConversionException"/>
/// with all errors concatenated.
/// </remarks>
public static partial class JsonSchemaToGbnf
{
    /// <summary>
    /// Compile a JSON-Schema string to a GBNF grammar. Equivalent to
    /// <c>llama.cpp</c>'s <c>json_schema_to_grammar(schema)</c>.
    /// </summary>
    public static string Convert(string jsonSchema, ConversionOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(jsonSchema))
        {
            throw new ArgumentException("Schema must not be empty.", nameof(jsonSchema));
        }
        using var doc = JsonDocument.Parse(jsonSchema);
        return Convert(doc.RootElement, options);
    }

    /// <summary>
    /// Compile an already-parsed JSON-Schema to a GBNF grammar.
    /// </summary>
    public static string Convert(JsonElement schema, ConversionOptions? options = null)
    {
        var opts = options ?? new ConversionOptions();
        var converter = new SchemaConverter(opts.Dotall);
        var working = CloneAsMutable(schema);
        converter.ResolveRefs(working, string.Empty);
        converter.Visit(working, string.Empty);
        converter.CheckErrors();
        return converter.FormatGrammar();
    }

    /// <summary>Options that change how the schema is compiled.</summary>
    public sealed record ConversionOptions(bool Dotall = false);

    // ============================================================
    // Shared constants & small helpers
    // ============================================================

    internal const string SpaceRule = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

    /// <summary>One built-in rule. <see cref="Deps"/> lists other built-ins that must also be emitted.</summary>
    internal sealed record BuiltinRule(string Content, IReadOnlyList<string> Deps);

    internal static readonly IReadOnlyDictionary<string, BuiltinRule> PrimitiveRules =
        new Dictionary<string, BuiltinRule>
        {
            ["boolean"]       = new("(\"true\" | \"false\") space", Array.Empty<string>()),
            ["decimal-part"]  = new("[0-9]{1,16}", Array.Empty<string>()),
            ["integral-part"] = new("[0] | [1-9] [0-9]{0,15}", Array.Empty<string>()),
            ["number"]        = new("(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space", new[] { "integral-part", "decimal-part" }),
            ["integer"]       = new("(\"-\"? integral-part) space", new[] { "integral-part" }),
            ["value"]         = new("object | array | string | number | boolean | null", new[] { "object", "array", "string", "number", "boolean", "null" }),
            ["object"]        = new("\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space", new[] { "string", "value" }),
            ["array"]         = new("\"[\" space ( value (\",\" space value)* )? \"]\" space", new[] { "value" }),
            ["uuid"]          = new("\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space", Array.Empty<string>()),
            ["char"]          = new("[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", Array.Empty<string>()),
            ["string"]        = new("\"\\\"\" char* \"\\\"\" space", new[] { "char" }),
            ["null"]          = new("\"null\" space", Array.Empty<string>()),
        };

    internal static readonly IReadOnlyDictionary<string, BuiltinRule> StringFormatRules =
        new Dictionary<string, BuiltinRule>
        {
            ["date"]             = new("[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", Array.Empty<string>()),
            ["time"]             = new("([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", Array.Empty<string>()),
            ["date-time"]        = new("date \"T\" time", new[] { "date", "time" }),
            ["date-string"]      = new("\"\\\"\" date \"\\\"\" space", new[] { "date" }),
            ["time-string"]      = new("\"\\\"\" time \"\\\"\" space", new[] { "time" }),
            ["date-time-string"] = new("\"\\\"\" date-time \"\\\"\" space", new[] { "date-time" }),
        };

    private static readonly HashSet<string> ReservedNames = BuildReservedNames();

    private static HashSet<string> BuildReservedNames()
    {
        var s = new HashSet<string>(StringComparer.Ordinal) { "root" };
        foreach (var k in PrimitiveRules.Keys) s.Add(k);
        foreach (var k in StringFormatRules.Keys) s.Add(k);
        return s;
    }

    internal static bool IsReservedName(string name) => ReservedNames.Contains(name);

    internal static readonly Regex InvalidRuleChars = new("[^a-zA-Z0-9-]+", RegexOptions.Compiled);
    internal static readonly Regex GrammarLiteralEscape = new("[\r\n\"\\\\]", RegexOptions.Compiled);
    internal static readonly IReadOnlyDictionary<char, string> GrammarLiteralEscapes =
        new Dictionary<char, string>
        {
            ['\r'] = @"\r", ['\n'] = @"\n", ['"'] = "\\\"",
            ['-'] = @"\-",  [']'] = @"\]",  ['\\'] = @"\\",
        };

    internal static readonly HashSet<char> NonLiteralSet =
        new() { '|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?' };

    internal static readonly HashSet<char> EscapedInRegexpsButNotInLiterals =
        new() { '^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?' };

    /// <summary>
    /// Wrap <paramref name="literal"/> in double-quotes and escape the
    /// characters GBNF requires (CR/LF/quote/backslash).
    /// </summary>
    internal static string FormatLiteral(string literal)
    {
        var escaped = GrammarLiteralEscape.Replace(literal, m =>
            GrammarLiteralEscapes.TryGetValue(m.Value[0], out var rep) ? rep : m.Value);
        return "\"" + escaped + "\"";
    }

    /// <summary>
    /// <c>{min_items,max_items}</c> repetition helper. Emits <c>?</c>,
    /// <c>*</c>, <c>+</c> shorthands when they fit, otherwise a full
    /// <c>{n,m}</c> range. With a <paramref name="separator"/>, expands
    /// to the "item (sep item){min-1,max-1}" pattern since GBNF doesn't
    /// have a native separator operator.
    /// </summary>
    internal static string BuildRepetition(string item, int minItems, int maxItems, string separator = "")
    {
        bool hasMax = maxItems != int.MaxValue;
        if (maxItems == 0) return string.Empty;
        if (minItems == 0 && maxItems == 1) return item + "?";

        if (separator.Length == 0)
        {
            if (minItems == 1 && !hasMax) return item + "+";
            if (minItems == 0 && !hasMax) return item + "*";
            return item + "{" + minItems + "," + (hasMax ? maxItems.ToString() : string.Empty) + "}";
        }

        // Separator present — expand manually. "item (sep item){min-1, max-1}".
        var inner = BuildRepetition(
            "(" + separator + " " + item + ")",
            minItems == 0 ? 0 : minItems - 1,
            hasMax ? maxItems - 1 : maxItems);
        var result = item + " " + inner;
        if (minItems == 0) result = "(" + result + ")?";
        return result;
    }

    /// <summary>
    /// Deep-clone a <see cref="JsonElement"/> into a mutable tree built
    /// out of <see cref="Dictionary{TKey,TValue}"/> / <see cref="List{T}"/>
    /// / primitives. The C++ reference mutates schemas in place during
    /// <c>$ref</c> resolution; the easiest way to match that is to work
    /// on a tree we own.
    /// </summary>
    private static object? CloneAsMutable(JsonElement el)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.Object:
                var obj = new OrderedJsonObject();
                foreach (var p in el.EnumerateObject())
                {
                    obj[p.Name] = CloneAsMutable(p.Value);
                }
                return obj;
            case JsonValueKind.Array:
                var arr = new List<object?>();
                foreach (var v in el.EnumerateArray())
                {
                    arr.Add(CloneAsMutable(v));
                }
                return arr;
            case JsonValueKind.String:   return el.GetString();
            case JsonValueKind.Number:
                return el.TryGetInt64(out var i) ? (object)i : el.GetDouble();
            case JsonValueKind.True:     return true;
            case JsonValueKind.False:    return false;
            case JsonValueKind.Null:     return null;
            default:                     return null;
        }
    }

    /// <summary>
    /// Minimal ordered-map type — schema key order matters for
    /// reproducible grammar output, and <see cref="Dictionary{TKey,TValue}"/>
    /// doesn't guarantee insertion order. We only need the narrow slice
    /// of dictionary ops that the converter uses.
    /// </summary>
    internal sealed class OrderedJsonObject : System.Collections.IEnumerable
    {
        private readonly List<string> _keys = new();
        private readonly Dictionary<string, object?> _map = new(StringComparer.Ordinal);

        public int Count => _keys.Count;

        public object? this[string key]
        {
            get => _map.TryGetValue(key, out var v) ? v : null;
            set
            {
                if (!_map.ContainsKey(key)) _keys.Add(key);
                _map[key] = value;
            }
        }

        public bool ContainsKey(string key) => _map.ContainsKey(key);
        public bool TryGetValue(string key, out object? value) => _map.TryGetValue(key, out value);

        public IEnumerable<KeyValuePair<string, object?>> Items()
        {
            foreach (var k in _keys) yield return new KeyValuePair<string, object?>(k, _map[k]);
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => Items().GetEnumerator();
    }
}

/// <summary>
/// Raised when <see cref="JsonSchemaToGbnf.Convert(string, JsonSchemaToGbnf.ConversionOptions?)"/>
/// encounters one or more schema-level errors (unknown types, unbalanced
/// regex, circular refs). The <see cref="Exception.Message"/> holds every
/// accumulated error joined by newlines.
/// </summary>
public sealed class JsonSchemaConversionException : Exception
{
    public JsonSchemaConversionException(string message) : base(message) { }
}
