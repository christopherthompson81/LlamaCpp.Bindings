namespace LlamaCpp.Bindings.LlamaChat.Models;

public sealed record ModelLoadSettings
{
    public string ModelPath { get; init; } = string.Empty;
    public int GpuLayerCount { get; init; } = -1;
    public uint ContextSize { get; init; } = 4096;
    public uint LogicalBatchSize { get; init; } = 512;
    public uint PhysicalBatchSize { get; init; } = 512;
    public bool UseMmap { get; init; } = true;
    public bool UseMlock { get; init; } = false;
    public bool OffloadKQV { get; init; } = true;
    public LlamaFlashAttention FlashAttention { get; init; } = LlamaFlashAttention.Auto;
}
