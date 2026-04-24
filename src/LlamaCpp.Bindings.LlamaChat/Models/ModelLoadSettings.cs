namespace LlamaCpp.Bindings.LlamaChat.Models;

public sealed record ModelLoadSettings
{
    public string ModelPath { get; init; } = string.Empty;

    /// <summary>
    /// Path to a matching <c>mmproj-*.gguf</c> (multimodal projector). When set
    /// and pointing at an existing file, the session loads an
    /// <see cref="MtmdContext"/> alongside the text model and images attached
    /// to a user turn get spliced into the prompt via mtmd's prefill path.
    /// Empty = text-only mode.
    /// </summary>
    public string MmprojPath { get; init; } = string.Empty;

    /// <summary>
    /// Force the vision encoder (CLIP) onto the CPU. Safer fallback for
    /// models whose CUDA CLIP graph is known-buggy (notably Qwen3-VL as of
    /// b8620 — see docs/mtmd_investigation.md). Adds ~1–5 s per image but
    /// avoids a hard process-killing segfault during image encode.
    /// Default: false (use GPU — matches mtmd's own default).
    /// </summary>
    public bool MmprojOnCpu { get; init; } = false;

    /// <summary>
    /// Minimum image tokens for dynamic-resolution vision models. null = let
    /// mtmd pick from model metadata. Qwen-VL models warn that grounding
    /// accuracy degrades below 1024 — set this to 1024 if the replies read
    /// like the model couldn't see the image clearly.
    /// </summary>
    public int? MmprojImageMinTokens { get; init; }

    public int GpuLayerCount { get; init; } = -1;
    public uint ContextSize { get; init; } = 4096;
    public uint LogicalBatchSize { get; init; } = 512;
    public uint PhysicalBatchSize { get; init; } = 512;
    public bool UseMmap { get; init; } = true;
    public bool UseMlock { get; init; } = false;
    public bool OffloadKQV { get; init; } = true;
    public LlamaFlashAttention FlashAttention { get; init; } = LlamaFlashAttention.Auto;

    /// <summary>
    /// KV cache element type for the K tensor. Default <c>F16</c> matches
    /// llama.cpp's native default. Quantized choices (Q8_0, Q5_0/1, Q4_0/1,
    /// IQ4_NL) require <see cref="FlashAttention"/> to be effectively enabled
    /// on the compute path — the UI auto-flips <see cref="FlashAttention"/>
    /// to <see cref="LlamaFlashAttention.Enabled"/> when the user picks a
    /// quantized KV type from the Auto default.
    /// </summary>
    public LlamaKvCacheType KvCacheTypeK { get; init; } = LlamaKvCacheType.F16;

    /// <summary>See <see cref="KvCacheTypeK"/>.</summary>
    public LlamaKvCacheType KvCacheTypeV { get; init; } = LlamaKvCacheType.F16;
}
