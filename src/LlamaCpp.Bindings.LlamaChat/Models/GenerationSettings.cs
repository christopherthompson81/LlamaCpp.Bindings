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

    /// <summary>
    /// Extract the Qwen3-ASR preamble (<c>language &lt;LANG&gt;&lt;asr_text&gt;…</c>) from
    /// the content stream. When enabled, the detected language is surfaced as
    /// a separate "language chip" on the message VM and the transcription
    /// flows into the normal content channel starting on its own line. When
    /// disabled (default — the common case for non-ASR models), the extractor
    /// is bypassed and nothing gets touched. Turn on for profiles loading
    /// Qwen3-ASR / Qwen3-Omni audio replies.
    /// </summary>
    public bool ExtractAsrTranscript { get; init; } = false;
}
