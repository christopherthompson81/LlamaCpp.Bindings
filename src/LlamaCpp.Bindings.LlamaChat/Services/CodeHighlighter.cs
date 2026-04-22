using System;
using System.Collections.Generic;
using System.Threading;
using ColorCode;
using ColorCode.Common;
using ColorCode.Compilation;
using ColorCode.Parsing;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Tokenises a source string into <see cref="Token"/> spans using
/// ColorCode.Core's <see cref="LanguageParser"/>. Each token carries its
/// ColorCode scope name (e.g. <c>"Keyword"</c>, <c>"String"</c>,
/// <c>"Comment"</c>) — the renderer maps those to theme brushes so syntax
/// highlighting tracks the Light/Dark variant. Unknown languages fall
/// through to a single plain-text token — the code block still renders,
/// just without colour.
/// </summary>
public static class CodeHighlighter
{
    public readonly record struct Token(string Text, string? Scope);

    // Parser built once and reused across all calls. LanguageParser is
    // thread-safe for reads of already-compiled languages; the underlying
    // CompiledLanguages cache has a ReaderWriterLockSlim.
    private static readonly LanguageParser _parser = BuildParser();

    private static LanguageParser BuildParser()
    {
        var repo = new LanguageRepository(new Dictionary<string, ILanguage>());
        foreach (var lang in Languages.All) repo.Load(lang);
        var compiler = new LanguageCompiler(
            new Dictionary<string, CompiledLanguage>(),
            new ReaderWriterLockSlim());
        return new LanguageParser(compiler, repo);
    }

    public static IReadOnlyList<Token> Highlight(string source, string? languageHint)
    {
        if (string.IsNullOrEmpty(source))
            return Array.Empty<Token>();

        var language = ResolveLanguage(languageHint);
        if (language is null)
            return new[] { new Token(source, null) };

        var tokens = new List<Token>();
        try
        {
            _parser.Parse(source, language, (text, scopes) =>
            {
                var scope = scopes is { Count: > 0 } ? scopes[0].Name : null;
                tokens.Add(new Token(text, scope));
            });
        }
        catch
        {
            // If ColorCode chokes on unusual input, fall back to plain text
            // rather than refusing to render the block.
            return new[] { new Token(source, null) };
        }

        return tokens.Count > 0 ? tokens : new[] { new Token(source, null) };
    }

    /// <summary>
    /// Map a fenced code block's info string to a ColorCode language.
    /// Covers the languages ColorCode.Core actually ships
    /// (`ColorCode.Languages`). Python, Haskell, Markdown, MATLAB, Fortran
    /// are available alongside the web/.NET/SQL set. Rust, Go, Ruby, YAML,
    /// Kotlin, Swift, bash etc. fall through to unhighlighted text until we
    /// either bolt on TextMateSharp or add hand-written rules.
    /// </summary>
    private static ILanguage? ResolveLanguage(string? hint)
    {
        if (string.IsNullOrWhiteSpace(hint)) return null;
        return hint.Trim().ToLowerInvariant() switch
        {
            "cs" or "c#" or "csharp" => Languages.CSharp,
            "cpp" or "c++" or "cxx" or "cc" or "hpp" or "c" or "h" => Languages.Cpp,
            "css" => Languages.Css,
            "fs" or "fsharp" or "f#" => Languages.FSharp,
            "html" or "htm" => Languages.Html,
            "java" => Languages.Java,
            "js" or "javascript" or "json" or "jsonc" => Languages.JavaScript,
            "py" or "python" => Languages.Python,
            "php" => Languages.Php,
            "ps1" or "pwsh" or "powershell" => Languages.PowerShell,
            "sql" => Languages.Sql,
            "ts" or "typescript" or "tsx" => Languages.Typescript,
            "xml" or "svg" or "xaml" or "axaml" => Languages.Xml,
            "md" or "markdown" => Languages.Markdown,
            "hs" or "haskell" => Languages.Haskell,
            "matlab" => Languages.MATLAB,
            "vb" or "vbnet" or "vb.net" => Languages.VbDotNet,
            "fortran" => Languages.Fortran,
            _ => null,
        };
    }
}
