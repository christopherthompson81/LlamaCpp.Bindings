namespace LlamaCpp.Bindings;

/// <summary>
/// A GBNF grammar source plus its start rule. Consumed by
/// <see cref="LlamaSamplerBuilder.WithGrammar"/> to constrain generation
/// to strings accepted by the grammar — the canonical use case is forcing
/// JSON-valid output, "function call" object shapes, SQL restricted to
/// safe queries, etc.
/// </summary>
/// <param name="GbnfSource">
/// The grammar as a string in llama.cpp's GBNF dialect. See
/// https://github.com/ggml-org/llama.cpp/tree/master/grammars for examples.
/// </param>
/// <param name="StartRuleName">
/// Name of the start symbol in the grammar (e.g., <c>"root"</c>).
/// </param>
public readonly record struct LlamaGrammar(string GbnfSource, string StartRuleName = "root")
{
    /// <summary>
    /// Shortcut for llama.cpp's bundled JSON grammar — any valid JSON value.
    /// Useful for coercing a model into "JSON mode" without writing a bespoke
    /// grammar.
    /// </summary>
    /// <remarks>
    /// This is the JSON grammar from llama.cpp's
    /// <c>grammars/json.gbnf</c>. If the upstream grammar evolves, this
    /// constant stays pinned to the version that shipped with the binding's
    /// pinned llama.cpp commit.
    /// </remarks>
    public static LlamaGrammar Json { get; } = new(
        GbnfSource: """
            root   ::= object
            value  ::= object | array | string | number | ("true" | "false" | "null") ws

            object ::=
              "{" ws (
                        string ":" ws value
                ("," ws string ":" ws value)*
              )? "}" ws

            array  ::=
              "[" ws (
                        value
                ("," ws value)*
              )? "]" ws

            string ::=
              "\"" (
                [^"\\\x7F\x00-\x1F] |
                "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
              )* "\"" ws

            number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
            ws ::= | " " | "\n" [ \t]{0,20}
            """,
        StartRuleName: "root");
}

/// <summary>
/// Lazy grammar: only begins constraining output once a trigger pattern or
/// trigger token is seen. Useful for "tool calling" flows where the model
/// should produce free-form text until it decides to emit a structured
/// call.
/// </summary>
/// <param name="Grammar">The underlying grammar.</param>
/// <param name="TriggerPatterns">
/// Regex-like patterns matched from the start of generation. The grammar
/// engages starting from the first match group.
/// </param>
/// <param name="TriggerTokens">
/// Token ids that trigger the grammar. The grammar is fed content starting
/// from (and including) the trigger token.
/// </param>
public readonly record struct LlamaLazyGrammar(
    LlamaGrammar Grammar,
    IReadOnlyList<string> TriggerPatterns,
    IReadOnlyList<int> TriggerTokens);
