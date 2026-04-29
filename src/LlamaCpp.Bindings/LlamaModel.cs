using System.Runtime.InteropServices;
using System.Text;
using LlamaCpp.Bindings.Native;
using LlamaCpp.Bindings.Native.SafeHandles;

namespace LlamaCpp.Bindings;

/// <summary>
/// A loaded GGUF model. One <see cref="LlamaModel"/> can back many
/// <see cref="LlamaContext"/> instances, but every context must be disposed
/// before the model is disposed.
/// </summary>
public sealed class LlamaModel : IDisposable
{
    private readonly SafeLlamaModelHandle _handle;
    private bool _disposed;

    /// <summary>Path the model was loaded from — purely informational.</summary>
    public string ModelPath { get; }

    /// <summary>Training context size (n_ctx_train) baked into the GGUF.</summary>
    public int TrainingContextSize { get; }

    /// <summary>Hidden-state embedding dimension (n_embd).</summary>
    public int EmbeddingSize { get; }

    /// <summary>Transformer layer count (n_layer).</summary>
    public int LayerCount { get; }

    /// <summary>Input-side embedding dimension (n_embd_inp). Equals <see cref="EmbeddingSize"/> on most decoder-only models.</summary>
    public int EmbeddingInputSize { get; }

    /// <summary>Output-side embedding dimension (n_embd_out). Equals <see cref="EmbeddingSize"/> on most decoder-only models.</summary>
    public int EmbeddingOutputSize { get; }

    /// <summary>Number of attention heads.</summary>
    public int AttentionHeadCount { get; }

    /// <summary>Number of KV heads. For GQA models this is smaller than <see cref="AttentionHeadCount"/>.</summary>
    public int KvHeadCount { get; }

    /// <summary>Sliding-window-attention window size, or 0 if the model uses dense attention.</summary>
    public int SlidingWindowSize { get; }

    /// <summary>RoPE positional-encoding variant.</summary>
    public LlamaRopeType RopeType { get; }

    /// <summary>
    /// Convenience getter for models that use rotary position embeddings —
    /// true for anything except <see cref="LlamaRopeType.None"/>.
    /// </summary>
    public bool UsesRotaryEmbeddings => RopeType != LlamaRopeType.None;

    /// <summary>RoPE frequency scaling factor baked into the training config.</summary>
    public float TrainingRopeFreqScale { get; }

    /// <summary>Total size on disk in bytes (what <c>llama_model_size</c> reports).</summary>
    public long SizeInBytes { get; }

    /// <summary>Total parameter count across all tensors.</summary>
    public long ParameterCount { get; }

    /// <summary>True if the model contains an encoder tower (e.g. T5-style).</summary>
    public bool HasEncoder { get; }

    /// <summary>True if the model contains a decoder tower. Most chat models are decoder-only.</summary>
    public bool HasDecoder { get; }

    /// <summary>Model uses recurrent state (Mamba / RWKV / similar) rather than attention.</summary>
    public bool IsRecurrent { get; }

    /// <summary>Model combines attention with recurrent state (hybrid SSM/attention).</summary>
    public bool IsHybrid { get; }

    /// <summary>Model is a diffusion-style generator rather than autoregressive.</summary>
    public bool IsDiffusion { get; }

    /// <summary>Decoder start token for encoder-decoder models, or <c>LLAMA_TOKEN_NULL</c> (-1) if not applicable.</summary>
    public int? DecoderStartToken { get; }

    /// <summary>
    /// Number of classifier output heads for classifier/reranker models.
    /// 0 for ordinary generative models.
    /// </summary>
    public int ClassifierOutputCount { get; }

    /// <summary>
    /// Human-readable architecture description emitted by
    /// <c>llama_model_desc</c> (e.g., <c>"llama 7B Q4_K_M"</c>).
    /// </summary>
    public string Description { get; }

    private IReadOnlyDictionary<string, string>? _metadata;

    /// <summary>
    /// Model metadata (the GGUF key/value dictionary). Materialised on first
    /// access and cached — typical models have 20-50 entries, a few KB total.
    /// Keys follow the GGUF namespacing convention (<c>general.name</c>,
    /// <c>tokenizer.ggml.bos_token_id</c>, etc.).
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata
    {
        get
        {
            EnsureNotDisposed();
            return _metadata ??= LoadMetadata();
        }
    }

    /// <summary>
    /// Look up one metadata value by exact key. More efficient than enumerating
    /// <see cref="Metadata"/> when you only want a single entry. Returns null
    /// if the key is not present.
    /// </summary>
    public unsafe string? GetMetadata(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        EnsureNotDisposed();
        var raw = Handle.DangerousHandle;
        return ProbeUtf8(
            (byte* buf, nuint sz) =>
                NativeMethods.llama_model_meta_val_str(raw, key, buf, sz));
    }

    /// <summary>
    /// Label string for classifier output index <paramref name="index"/>, or
    /// null if the model has no label for that index (or isn't a classifier).
    /// </summary>
    public string? GetClassifierLabel(int index)
    {
        EnsureNotDisposed();
        if (index < 0 || index >= ClassifierOutputCount) return null;
        var ptr = NativeMethods.llama_model_cls_label(Handle.DangerousHandle, (uint)index);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    /// <summary>
    /// Raw vocab pointer. Consumed by the <c>llama_vocab_*</c> family of
    /// functions (tokenize, BOS, EOS, is-EOG). Not exposed as a safe handle
    /// because llama.cpp treats the vocab as owned-by-model — freeing the
    /// model frees the vocab.
    /// </summary>
    internal IntPtr VocabHandle { get; }

    private LlamaVocab? _vocab;

    /// <summary>
    /// Tokenizer + special-token registry. Materialised lazily on first access.
    /// Invalidated when this model is disposed.
    /// </summary>
    public LlamaVocab Vocab
    {
        get
        {
            EnsureNotDisposed();
            return _vocab ??= new LlamaVocab(this, VocabHandle);
        }
    }

    /// <summary>
    /// Returns the chat template embedded in the GGUF metadata, or null if the
    /// model ships without one. Typically a Jinja-style string consumed by
    /// <see cref="LlamaChatTemplate.Apply"/>. The return is a view into the
    /// model's metadata — it must not outlive this <see cref="LlamaModel"/>.
    /// </summary>
    /// <param name="name">
    /// Named template to fetch (some GGUFs ship multiple). <c>null</c> returns
    /// the default.
    /// </param>
    public string? GetChatTemplate(string? name = null)
    {
        EnsureNotDisposed();
        var ptr = NativeMethods.llama_model_chat_template(Handle.DangerousHandle, name);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    public unsafe LlamaModel(string path, LlamaModelParameters? parameters = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        LlamaBackend.EnsureInitialized();

        // Pin() bundles the native struct with any unmanaged side
        // allocations (devices + tensor_split arrays) so they outlive the
        // load call. The using block guarantees the buffers are freed
        // even if load throws.
        using var pinned = (parameters ?? LlamaModelParameters.Default()).Pin();
        var raw = NativeMethods.llama_model_load_from_file(path, pinned.Native);
        if (raw == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_model_load_from_file),
                $"Failed to load model from '{path}'. Check the native log output for details.");
        }

        _handle = SafeLlamaModelHandle.FromUnsafeHandle(raw);
        ModelPath = path;

        VocabHandle = NativeMethods.llama_model_get_vocab(raw);
        if (VocabHandle == IntPtr.Zero)
        {
            _handle.Dispose();
            throw new LlamaException(
                nameof(NativeMethods.llama_model_get_vocab),
                "llama_model_get_vocab returned NULL — model loaded but has no vocab.");
        }

        TrainingContextSize    = NativeMethods.llama_model_n_ctx_train(raw);
        EmbeddingSize          = NativeMethods.llama_model_n_embd(raw);
        EmbeddingInputSize     = NativeMethods.llama_model_n_embd_inp(raw);
        EmbeddingOutputSize    = NativeMethods.llama_model_n_embd_out(raw);
        LayerCount             = NativeMethods.llama_model_n_layer(raw);
        AttentionHeadCount     = NativeMethods.llama_model_n_head(raw);
        KvHeadCount            = NativeMethods.llama_model_n_head_kv(raw);
        SlidingWindowSize      = NativeMethods.llama_model_n_swa(raw);
        RopeType               = (LlamaRopeType)NativeMethods.llama_model_rope_type(raw);
        TrainingRopeFreqScale  = NativeMethods.llama_model_rope_freq_scale_train(raw);
        SizeInBytes            = (long)NativeMethods.llama_model_size(raw);
        ParameterCount         = (long)NativeMethods.llama_model_n_params(raw);
        HasEncoder             = NativeMethods.llama_model_has_encoder(raw);
        HasDecoder             = NativeMethods.llama_model_has_decoder(raw);
        IsRecurrent            = NativeMethods.llama_model_is_recurrent(raw);
        IsHybrid               = NativeMethods.llama_model_is_hybrid(raw);
        IsDiffusion            = NativeMethods.llama_model_is_diffusion(raw);
        ClassifierOutputCount  = (int)NativeMethods.llama_model_n_cls_out(raw);

        var decoderStart = NativeMethods.llama_model_decoder_start_token(raw);
        DecoderStartToken = decoderStart == LlamaVocab.LLAMA_TOKEN_NULL ? null : decoderStart;

        Description = ProbeUtf8(
            (byte* buf, nuint sz) => NativeMethods.llama_model_desc(raw, buf, sz))
            ?? string.Empty;
    }

    private unsafe IReadOnlyDictionary<string, string> LoadMetadata()
    {
        var count = NativeMethods.llama_model_meta_count(Handle.DangerousHandle);
        if (count <= 0) return new Dictionary<string, string>(0);

        var dict = new Dictionary<string, string>(count, StringComparer.Ordinal);
        var raw = Handle.DangerousHandle;
        for (int i = 0; i < count; i++)
        {
            int idx = i; // capture for the closures below
            var key = ProbeUtf8(
                (byte* buf, nuint sz) => NativeMethods.llama_model_meta_key_by_index(raw, idx, buf, sz));
            if (string.IsNullOrEmpty(key)) continue;
            var val = ProbeUtf8(
                (byte* buf, nuint sz) => NativeMethods.llama_model_meta_val_str_by_index(raw, idx, buf, sz))
                ?? string.Empty;
            dict[key] = val;
        }
        return dict;
    }

    /// <summary>
    /// Runs a native "probe then fill" buffer call twice: once with a 256-byte
    /// stack buffer, and if the native side wanted more, once with a heap
    /// buffer of the exact size. Returns null if the probe returned a negative
    /// status (meaning the item isn't available).
    /// </summary>
    private unsafe delegate int ProbeCall(byte* buf, nuint size);

    private static unsafe string? ProbeUtf8(ProbeCall call)
    {
        const int StackBuf = 256;
        byte* stackPtr = stackalloc byte[StackBuf];
        int needed = call(stackPtr, StackBuf);
        if (needed < 0) return null;

        // llama.cpp returns the length *without* the null terminator. If the
        // buffer was enough, just decode. If not, retry with exactly the
        // reported size.
        if (needed <= StackBuf)
        {
            return Encoding.UTF8.GetString(stackPtr, needed);
        }
        var heap = new byte[needed + 1];
        fixed (byte* heapPtr = heap)
        {
            int written = call(heapPtr, (nuint)heap.Length);
            if (written < 0) return null;
            return Encoding.UTF8.GetString(heapPtr, written);
        }
    }

    internal SafeLlamaModelHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    internal void EnsureNotDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>
    /// Overwrite the data of an already-loaded tensor in this model.
    /// Locates the tensor by exact-match name in the model's
    /// <c>tensors_by_name</c> vector, then dispatches the bytes to the
    /// owning backend (CUDA / CPU / Metal / …) via
    /// <c>ggml_backend_tensor_set</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by the AdaptiveQuantization profile builder to apply per-cell
    /// ablations without reloading the model: the same F16 model stays
    /// resident in VRAM across ablations, and only the affected tensor's
    /// bytes are rewritten between PPL passes.
    /// </para>
    /// <para>
    /// <strong>Buffer-size constraint:</strong> the model's tensor
    /// allocation is fixed at load time by the source GGUF's tensor type.
    /// Writes must match the existing tensor's byte size — i.e., to ablate
    /// an F16 tensor through Q4_K, callers must round-trip the data
    /// (F16 → Q4_K → F16) so the bytes written are still F16-shaped.
    /// Writing fewer or more bytes than the tensor's allocated buffer
    /// silently corrupts neighbouring tensors.
    /// </para>
    /// <para>
    /// Bypasses llama.cpp's public API by reaching into the internal
    /// <c>llama_internal_get_tensor_map</c> symbol via the
    /// <c>libllamashim</c> shim. The shim is rebuilt against each pinned
    /// llama.cpp version (<c>tools/native-shims/build.sh</c>); failure to
    /// rebuild after a pin bump will surface as a missing-symbol
    /// <see cref="DllNotFoundException"/> on first call.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// No tensor with that name exists in the model, or the shim
    /// rejected the call (returned a non-zero status).
    /// </exception>
    public unsafe void SetTensorData(string tensorName, ReadOnlySpan<byte> data, long offset = 0)
    {
        ArgumentException.ThrowIfNullOrEmpty(tensorName);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        EnsureNotDisposed();

        var tensor = NativeMethods.llamashim_get_model_tensor(_handle.DangerousGetHandle(), tensorName);
        if (tensor == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' not found in model. Use {nameof(EnumerateTensorNames)} to inspect available names.");
        }

        fixed (byte* dataPtr = data)
        {
            int rc = NativeMethods.llamashim_set_tensor_data(
                tensor, dataPtr, (nuint)offset, (nuint)data.Length);
            if (rc != 0)
            {
                throw new InvalidOperationException(
                    $"llamashim_set_tensor_data returned {rc} for tensor '{tensorName}'.");
            }
        }
    }

    /// <summary>
    /// Compute a content fingerprint (SHA-256) of an already-loaded
    /// tensor's GPU bytes. Diagnostic helper for the in-place ablator
    /// path — lets callers confirm "the bytes I wrote are still there"
    /// without serializing the whole tensor.
    /// </summary>
    /// <remarks>
    /// Reads the full tensor buffer via <see cref="GetTensorData"/>
    /// (one D2H of the tensor's <c>ByteSize</c>) and hashes it.
    /// Cost is dominated by the d2h transfer.
    /// </remarks>
    public string FingerprintTensor(string tensorName)
    {
        ArgumentException.ThrowIfNullOrEmpty(tensorName);
        EnsureNotDisposed();

        var tensor = NativeMethods.llamashim_get_model_tensor(_handle.DangerousGetHandle(), tensorName);
        if (tensor == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' not found in model.");
        }

        // Pull the tensor's byte size from the shim's get-by-index path —
        // we don't have a direct "tensor.ByteSize" accessor, so we rely
        // on the caller having a LlamaGgufFile or pass it explicitly.
        // Simplest portable solution: take size from a one-time GgufFile
        // open. Callers who already have an LlamaGgufFile should use the
        // <see cref="FingerprintTensorWithSize"/> overload instead to
        // avoid re-reading the GGUF header.
        var ggufFile = LlamaGgufFile.Open(ModelPath);
        var info = ggufFile.Tensors.FirstOrDefault(t => t.Name == tensorName)
            ?? throw new InvalidOperationException(
                $"Tensor '{tensorName}' missing from GGUF header.");
        return FingerprintTensorWithSize(tensorName, info.ByteSize);
    }

    /// <summary>
    /// Faster variant of <see cref="FingerprintTensor"/> when the
    /// caller already knows the tensor's byte size (e.g. from a
    /// previously-opened <see cref="LlamaGgufFile"/>) — avoids the
    /// per-call GGUF header reparse.
    /// </summary>
    public string FingerprintTensorWithSize(string tensorName, long byteSize)
    {
        ArgumentException.ThrowIfNullOrEmpty(tensorName);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteSize);
        EnsureNotDisposed();

        var buf = new byte[byteSize];
        GetTensorData(tensorName, buf);
        var hash = System.Security.Cryptography.SHA256.HashData(buf);
        return Convert.ToHexString(hash);
    }

    /// <summary>
    /// Read the raw bytes of an already-loaded tensor's buffer.
    /// Symmetric companion to <see cref="SetTensorData"/>; primarily
    /// used to verify that a prior write actually landed in GPU memory
    /// (i.e., distinguishing "the write took effect" from "the API
    /// returned success but nothing changed").
    /// </summary>
    /// <exception cref="InvalidOperationException">No tensor with that name, or shim returned non-zero status.</exception>
    public unsafe void GetTensorData(string tensorName, Span<byte> destination, long offset = 0)
    {
        ArgumentException.ThrowIfNullOrEmpty(tensorName);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        EnsureNotDisposed();

        var tensor = NativeMethods.llamashim_get_model_tensor(_handle.DangerousGetHandle(), tensorName);
        if (tensor == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                $"Tensor '{tensorName}' not found in model.");
        }

        fixed (byte* dataPtr = destination)
        {
            int rc = NativeMethods.llamashim_get_tensor_data(
                tensor, dataPtr, (nuint)offset, (nuint)destination.Length);
            if (rc != 0)
            {
                throw new InvalidOperationException(
                    $"llamashim_get_tensor_data returned {rc} for tensor '{tensorName}'.");
            }
        }
    }

    /// <summary>
    /// Enumerate every tensor name in the model's
    /// <c>tensors_by_name</c> vector. Order is the load-time ordering
    /// (deterministic given a fixed GGUF + llama.cpp version).
    /// </summary>
    /// <remarks>
    /// Useful for sanity-checking <see cref="SetTensorData"/> targets and
    /// for building name → index caches. Materializes the names eagerly;
    /// for large models (10K+ tensors) consider caching the result.
    /// </remarks>
    public IReadOnlyList<string> EnumerateTensorNames()
    {
        EnsureNotDisposed();
        var handle = _handle.DangerousGetHandle();
        var count = (int)NativeMethods.llamashim_get_tensor_count(handle);
        var names = new List<string>(count);
        for (int i = 0; i < count; i++)
        {
            var ptr = NativeMethods.llamashim_get_tensor_name_at(handle, (nuint)i);
            if (ptr == IntPtr.Zero) break;
            var name = Marshal.PtrToStringUTF8(ptr);
            if (name is not null) names.Add(name);
        }
        return names;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}

/// <summary>
/// Variant of rotary position embedding a model was trained with. Matches
/// llama.cpp's <c>llama_rope_type</c>, with ggml-side values spelled out
/// explicitly.
/// </summary>
public enum LlamaRopeType
{
    None   = -1,
    Normal = 0,
    NeoX   = 2,
    MRope  = 8,
    Vision = 24,
    IMRope = 40,
}
