namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>
/// A named, persistable bundle of load + sampling + generation settings.
/// Users pick a profile to load; the profile's sampler settings stay live
/// for that session and can be edited from the Settings window.
/// </summary>
public sealed record ModelProfile
{
    public string Name { get; init; } = "New profile";
    public ModelLoadSettings Load { get; init; } = new();
    public SamplerSettings Sampler { get; init; } = new();
    public GenerationSettings Generation { get; init; } = new();
}
