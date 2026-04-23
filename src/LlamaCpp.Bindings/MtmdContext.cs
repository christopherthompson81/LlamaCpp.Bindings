using System.Runtime.InteropServices;
using System.Text;
using LlamaCpp.Bindings.Native;
using LlamaCpp.Bindings.Native.SafeHandles;

namespace LlamaCpp.Bindings;

/// <summary>
/// A multimodal projector loaded against an existing text model. Decodes
/// image (and, in a later phase, audio) bitmaps into embeddings the matching
/// text model can attend over.
/// </summary>
/// <remarks>
/// One <see cref="MtmdContext"/> is paired with exactly one <see cref="LlamaModel"/>.
/// Construction calls <c>mtmd_init_from_file</c> and, unless disabled, runs
/// a warmup encode pass that can take a couple of seconds on cold start.
/// Must be disposed before the paired <see cref="LlamaModel"/>.
/// </remarks>
public sealed class MtmdContext : IDisposable
{
    private readonly SafeMtmdContextHandle _handle;
    private readonly LlamaModel _model;
    private bool _disposed;

    /// <summary>True if this projector was built for vision (image) input.</summary>
    public bool SupportsVision { get; }

    /// <summary>True if this projector was built for audio input.</summary>
    public bool SupportsAudio { get; }

    /// <summary>Model requires a non-causal attention mask during image decode (e.g. Gemma-3 vision).</summary>
    public bool UsesNonCausalMask { get; }

    /// <summary>Model uses multi-dimensional RoPE for image positions (e.g. Qwen2.5-VL).</summary>
    public bool UsesMRope { get; }

    /// <summary>Audio sample rate expected by the encoder, or null if audio isn't supported.</summary>
    public int? AudioSampleRate { get; }

    /// <summary>
    /// The placeholder substring the user's prompt must contain for every
    /// bitmap. Each occurrence gets replaced with the expanded image/audio
    /// chunk during <c>mtmd_tokenize</c>. Typically <c>&lt;__media__&gt;</c>.
    /// </summary>
    public string DefaultMediaMarker { get; }

    public LlamaModel Model
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _model;
        }
    }

    internal SafeMtmdContextHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    public MtmdContext(LlamaModel model, string mmprojPath, MtmdContextParameters? parameters = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(mmprojPath);
        if (!File.Exists(mmprojPath))
            throw new FileNotFoundException($"mmproj file not found: {mmprojPath}", mmprojPath);

        LlamaBackend.EnsureInitialized();
        LlamaBackend.RebindMtmdLogSink();

        _model = model;
        var native = (parameters ?? new MtmdContextParameters()).ToNative();

        var raw = NativeMethods.mtmd_init_from_file(mmprojPath, model.Handle.DangerousHandle, native);
        if (raw == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.mtmd_init_from_file),
                $"Failed to load mtmd projector from '{mmprojPath}'. Common causes: mmproj " +
                $"doesn't match the text model family, or the model wasn't trained with vision/audio. " +
                $"Check the native log output.");
        }

        _handle = SafeMtmdContextHandle.FromUnsafeHandle(raw);

        SupportsVision    = NativeMethods.mtmd_support_vision(raw);
        SupportsAudio     = NativeMethods.mtmd_support_audio(raw);
        UsesNonCausalMask = NativeMethods.mtmd_decode_use_non_causal(raw);
        UsesMRope         = NativeMethods.mtmd_decode_use_mrope(raw);

        var rate = NativeMethods.mtmd_get_audio_sample_rate(raw);
        AudioSampleRate = rate < 0 ? null : rate;

        var markerPtr = NativeMethods.mtmd_default_marker();
        DefaultMediaMarker = markerPtr == IntPtr.Zero
            ? string.Empty
            : Marshal.PtrToStringUTF8(markerPtr) ?? string.Empty;
    }

    /// <summary>
    /// Tokenize a prompt containing <see cref="DefaultMediaMarker"/> occurrences
    /// paired with <paramref name="bitmaps"/>, then prefill the llama context
    /// via <c>mtmd_helper_eval_chunks</c>. Returns the new n_past (caller
    /// stores this and passes it to the generator for subsequent decode).
    /// </summary>
    /// <param name="llamaContext">Target decode context; must share the text model.</param>
    /// <param name="prompt">
    /// UTF-8 prompt text. The number of marker substrings must equal
    /// <paramref name="bitmaps"/>.Count.
    /// </param>
    /// <param name="bitmaps">Bitmaps to splice at the marker positions, in order.</param>
    /// <param name="nPast">Position in the KV cache to begin writing at.</param>
    /// <param name="seqId">Target sequence id (0 for single-sequence chat).</param>
    /// <param name="nBatch">Physical batch size for llama_decode calls inside the helper.</param>
    /// <param name="logitsLast">
    /// If true, the final token's logits are computed so the caller can sample
    /// immediately. Pass false for pure prefill.
    /// </param>
    /// <param name="addSpecial">Emit BOS if the model expects it (pass true only for the very first prefill).</param>
    /// <param name="parseSpecial">Honour special-token sequences like <c>&lt;|im_start|&gt;</c> in the prompt.</param>
    public Task<int> EvalPromptAsync(
        LlamaContext llamaContext,
        string prompt,
        IReadOnlyList<MtmdBitmap> bitmaps,
        int nPast,
        int seqId = 0,
        int nBatch = 512,
        bool logitsLast = true,
        bool addSpecial = false,
        bool parseSpecial = true,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(llamaContext);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(bitmaps);

        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!ReferenceEquals(llamaContext.Model, _model))
        {
            throw new ArgumentException(
                "LlamaContext must be built from the same LlamaModel this MtmdContext was attached to.",
                nameof(llamaContext));
        }

        // mtmd_helper_eval_chunks is documented as NOT thread-safe, so we hop
        // onto a background thread to honour the async contract without
        // multiplexing calls.
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return EvalPromptCore(llamaContext, prompt, bitmaps, nPast, seqId, nBatch,
                logitsLast, addSpecial, parseSpecial, cancellationToken);
        }, cancellationToken);
    }

    private unsafe int EvalPromptCore(
        LlamaContext llamaContext,
        string prompt,
        IReadOnlyList<MtmdBitmap> bitmaps,
        int nPast,
        int seqId,
        int nBatch,
        bool logitsLast,
        bool addSpecial,
        bool parseSpecial,
        CancellationToken cancellationToken)
    {
        var promptUtf8 = Encoding.UTF8.GetBytes(prompt + '\0');
        using var chunks = SafeMtmdInputChunksHandle.FromUnsafeHandle(
            NativeMethods.mtmd_input_chunks_init());
        if (chunks.IsInvalid)
        {
            throw new LlamaException(
                nameof(NativeMethods.mtmd_input_chunks_init),
                "mtmd_input_chunks_init returned NULL.");
        }

        var bitmapHandles = new IntPtr[bitmaps.Count];
        for (int i = 0; i < bitmaps.Count; i++)
        {
            bitmapHandles[i] = bitmaps[i].Handle.DangerousHandle;
        }

        fixed (byte* promptPtr = promptUtf8)
        fixed (IntPtr* bitmapPtr = bitmapHandles)
        {
            var text = new mtmd_input_text
            {
                text          = (IntPtr)promptPtr,
                add_special   = addSpecial,
                parse_special = parseSpecial,
            };

            var tokStatus = NativeMethods.mtmd_tokenize(
                _handle.DangerousHandle,
                chunks.DangerousHandle,
                in text,
                bitmapPtr,
                (nuint)bitmapHandles.Length);
            if (tokStatus != 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.mtmd_tokenize),
                    tokStatus,
                    tokStatus switch
                    {
                        1 => "mtmd_tokenize: number of media markers in prompt does not match bitmap count.",
                        2 => "mtmd_tokenize: image preprocessing failed. Check the native log.",
                        _ => $"mtmd_tokenize returned status {tokStatus}.",
                    });
            }

            cancellationToken.ThrowIfCancellationRequested();

            var evalStatus = NativeMethods.mtmd_helper_eval_chunks(
                _handle.DangerousHandle,
                llamaContext.Handle.DangerousHandle,
                chunks.DangerousHandle,
                nPast,
                seqId,
                nBatch,
                logitsLast,
                out var newNPast);
            if (evalStatus != 0)
            {
                throw new LlamaException(nameof(NativeMethods.mtmd_helper_eval_chunks), evalStatus);
            }
            return newNPast;
        }
    }

    /// <summary>
    /// Sum of positional advances across <paramref name="chunks"/> — only useful
    /// when the caller owns its own chunk list (advanced flows). Exposed as a
    /// helper in case test code wants to cross-check <c>EvalPromptAsync</c>'s
    /// n_past delta without going through eval.
    /// </summary>
    internal static int GetNPos(SafeMtmdInputChunksHandle chunks)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        return NativeMethods.mtmd_helper_get_n_pos(chunks.DangerousHandle);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
