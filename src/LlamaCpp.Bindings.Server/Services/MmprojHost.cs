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
        if (string.IsNullOrWhiteSpace(opts.MmprojPath))
        {
            _log.LogInformation(
                "Multimodal disabled: LlamaServer:MmprojPath not set. " +
                "Chat requests with image_url parts will return 400.");
            return;
        }
        if (!File.Exists(opts.MmprojPath))
        {
            throw new FileNotFoundException(
                $"LlamaServer:MmprojPath='{opts.MmprojPath}' does not exist.", opts.MmprojPath);
        }

        _log.LogInformation("Loading mmproj projector from {Path}", opts.MmprojPath);
        var mtmdParams = new MtmdContextParameters
        {
            UseGpu = opts.MmprojOnCpu is bool cpu ? !cpu : null,
            ImageMinTokens = opts.MmprojImageMinTokens,
            ImageMaxTokens = opts.MmprojImageMaxTokens,
        };
        _mtmd = new MtmdContext(modelHost.Model, opts.MmprojPath, mtmdParams);
        _log.LogInformation(
            "mmproj loaded: vision={Vision}, audio={Audio}, marker={Marker}",
            _mtmd.SupportsVision, _mtmd.SupportsAudio, _mtmd.DefaultMediaMarker);
    }

    public void Dispose() => _mtmd?.Dispose();
}
