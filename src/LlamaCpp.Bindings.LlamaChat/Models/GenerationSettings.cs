namespace LlamaCpp.Bindings.LlamaChat.Models;

public sealed record GenerationSettings
{
    public int MaxTokens { get; init; } = 1024;

    /// <summary>
    /// Reuse the portion of the KV cache that still matches the newly rendered prompt prefix
    /// rather than clearing and re-decoding from scratch each turn. Turn off to force a
    /// full rebuild (useful when debugging determinism).
    /// </summary>
    public bool ReusePromptPrefix { get; init; } = true;

    /// <summary>
    /// Extract <![CDATA[<think>...</think>]]> blocks and route them to <see cref="ChatTurn.Reasoning"/>
    /// instead of the main content. Hardcoded tag pair for v1; template-driven config is deferred.
    /// </summary>
    public bool ExtractReasoning { get; init; } = true;
}
