using LlamaCpp.Bindings.Server.Configuration;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Optional multimodal projector attached to the main
/// <see cref="LlamaModel"/>. Loaded on startup when
/// <see cref="ServerOptions.MmprojPath"/> is set; otherwise
/// <see cref="IsAvailable"/> is false and chat requests with image
/// parts return HTTP 400.
/// </summary>
/// <remarks>
/// <para>The MTMD native helper that actually does the image-encode +
/// llama_decode dance (<c>mtmd_helper_eval_chunks</c>) is documented as
/// not thread-safe; the binding's <see cref="MtmdContext.EvalPromptAsync"/>
/// already hops to a background thread, and that plus the main context's
/// own decode lock is enough to keep concurrent calls safe. We don't
/// pool or duplicate the mtmd context — one is sufficient for the
/// request rate a local server serves.</para>
///
/// <para>Bitmaps are created via <see cref="MtmdBitmap.FromBytes"/>
/// which auto-detects format (PNG, JPEG, BMP, etc., plus audio types
/// when the projector is audio-capable). Callers hand raw bytes; we
/// don't enforce a size cap here — <see cref="ChatContentExtractor"/>
/// is responsible for limiting individual request payloads.</para>
/// </remarks>
public sealed class MmprojHost : IDisposable
{
    private readonly ILogger<MmprojHost> _log;
    private readonly MtmdContext? _mtmd;

    public bool IsAvailable => _mtmd is not null;

    public MtmdContext? Context => _mtmd;

    public MmprojHost(IOptions<ServerOptions> options, ModelHost modelHost, ILogger<MmprojHost> log)
    {
        _log = log;
        var opts = options.Value;

        // Resolution order: explicit path wins; otherwise, when MmprojAuto
        // is on, look for a sibling mmproj-*.gguf next to the main model.
        // The auto-probe never throws — multiple matches or zero matches
        // both fall through to "no mmproj", and the operator can fix
        // their config explicitly.
        string? path = opts.MmprojPath;
        if (string.IsNullOrWhiteSpace(path) && opts.MmprojAuto)
        {
            path = TryProbeSiblingMmproj(opts.ModelPath, _log);
        }

        if (string.IsNullOrWhiteSpace(path))
        {
            _log.LogInformation(
                "Multimodal disabled: no mmproj path resolved (MmprojPath unset" +
                (opts.MmprojAuto ? ", MmprojAuto found no sibling mmproj-*.gguf" : "") +
                "). Chat requests with image_url parts will return 400.");
            return;
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"LlamaServer:MmprojPath='{path}' does not exist.", path);
        }

        _log.LogInformation("Loading mmproj projector from {Path}", path);
        var mtmdParams = new MtmdContextParameters
        {
            UseGpu = opts.MmprojOnCpu is bool cpu ? !cpu : null,
            ImageMinTokens = opts.MmprojImageMinTokens,
            ImageMaxTokens = opts.MmprojImageMaxTokens,
        };
        _mtmd = new MtmdContext(modelHost.Model, path, mtmdParams);
        _log.LogInformation(
            "mmproj loaded: vision={Vision}, audio={Audio}, marker={Marker}",
            _mtmd.SupportsVision, _mtmd.SupportsAudio, _mtmd.DefaultMediaMarker);
    }

    /// <summary>
    /// Look for a single sibling <c>mmproj-*.gguf</c> file next to the
    /// main model. Returns the path on a unique match; null on zero or
    /// multiple matches (the latter logs a warning so the operator can
    /// disambiguate by setting <see cref="ServerOptions.MmprojPath"/>
    /// explicitly).
    /// </summary>
    private static string? TryProbeSiblingMmproj(string modelPath, ILogger log)
    {
        if (string.IsNullOrWhiteSpace(modelPath)) return null;
        var dir = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;

        // Convention covers both naming patterns seen in the wild:
        // mmproj-MODEL.gguf (HuggingFace) and MODEL.mmproj.gguf.
        var matches = Directory.EnumerateFiles(dir, "mmproj-*.gguf").ToList();
        matches.AddRange(Directory.EnumerateFiles(dir, "*.mmproj.gguf"));
        // Deduplicate in case both globs catch the same file.
        matches = matches.Distinct().ToList();

        if (matches.Count == 0) return null;
        if (matches.Count > 1)
        {
            log.LogWarning(
                "MmprojAuto: found {Count} candidate mmproj files in {Dir} " +
                "({Files}). Set MmprojPath explicitly to disambiguate; auto-probe skipped.",
                matches.Count, dir, string.Join(", ", matches.Select(Path.GetFileName)));
            return null;
        }
        log.LogInformation("MmprojAuto: using sibling {Path}", matches[0]);
        return matches[0];
    }

    public void Dispose() => _mtmd?.Dispose();
}
