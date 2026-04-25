namespace LlamaCpp.Bindings.LlamaChat.Models;

/// <summary>Local in-process inference vs HTTP backend.</summary>
public enum ProfileKind { Local, Remote }

/// <summary>
/// HTTP backend configuration. Used when <see cref="ModelProfile.Kind"/> is
/// <see cref="ProfileKind.Remote"/>; ignored otherwise. The shape targets
/// OpenAI-compatible <c>/v1/chat/completions</c> servers (notably
/// LlamaCpp.Bindings.Server, llama.cpp's own server, vLLM, OpenAI).
/// </summary>
public sealed record RemoteSettings
{
    /// <summary>
    /// Server root, e.g. <c>http://localhost:8080</c>. The client appends
    /// <c>/v1/chat/completions</c>, <c>/v1/models</c>, etc. — supply only
    /// the scheme + host + port (+ optional path prefix), no trailing slash
    /// required.
    /// </summary>
    public string BaseUrl { get; init; } = "http://localhost:8080";

    /// <summary>
    /// Bearer token sent in the Authorization header. Null/empty = no
    /// header. Stored plaintext alongside the rest of the profile (matches
    /// the existing convention for app-settings / MCP server config).
    /// </summary>
    public string? ApiKey { get; init; }

    /// <summary>
    /// Server-side model id sent in the request body's <c>model</c> field.
    /// Populated by the Discover button via <c>GET /v1/models</c>; can be
    /// edited by hand if the server enumerates a different name than the
    /// caller wants to use.
    /// </summary>
    public string ModelId { get; init; } = "";
}

/// <summary>
/// A named, persistable bundle of load + sampling + generation settings.
/// Users pick a profile to load; the profile's sampler settings stay live
/// for that session and can be edited from the Settings window.
/// </summary>
public sealed record ModelProfile
{
    public string Name { get; init; } = "New profile";

    /// <summary>
    /// Selects the inference backend: <see cref="ProfileKind.Local"/> loads a
    /// GGUF in-process via <see cref="ModelLoadSettings"/>; <see cref="ProfileKind.Remote"/>
    /// drives an HTTP server via <see cref="RemoteSettings"/>. Default Local
    /// preserves the historical zero-config behaviour for existing profile files.
    /// </summary>
    public ProfileKind Kind { get; init; } = ProfileKind.Local;

    /// <summary>
    /// Prepended as a system-role turn at the start of every transcript sent
    /// to the model under this profile. Empty = no system message.
    /// </summary>
    public string SystemPrompt { get; init; } = string.Empty;

    /// <summary>Local backend load knobs. Ignored when <see cref="Kind"/> is Remote.</summary>
    public ModelLoadSettings Load { get; init; } = new();

    /// <summary>Remote backend connection. Ignored when <see cref="Kind"/> is Local.</summary>
    public RemoteSettings Remote { get; init; } = new();

    public SamplerSettings Sampler { get; init; } = new();
    public GenerationSettings Generation { get; init; } = new();
}
