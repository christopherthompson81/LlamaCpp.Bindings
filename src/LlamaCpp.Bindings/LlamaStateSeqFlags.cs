namespace LlamaCpp.Bindings;

/// <summary>
/// Flags controlling what portion of a per-sequence snapshot is captured or
/// restored via the <c>Ext</c> variants on <see cref="LlamaContext"/>.
/// Mirrors <c>llama_state_seq_flags</c>.
/// </summary>
[Flags]
public enum LlamaStateSeqFlags : uint
{
    /// <summary>Capture/restore the full sequence state. Default behavior.</summary>
    None = 0,

    /// <summary>
    /// Operate only on partial states — the sliding-window (SWA) portion of
    /// the KV cache for transformer models, or the recurrent state for
    /// models like Mamba. Mirrors <c>LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY</c>
    /// (same value as the legacy <c>SWA_ONLY</c> flag retained for
    /// backwards-compat in the pinned header).
    /// </summary>
    PartialOnly = 1,
}
