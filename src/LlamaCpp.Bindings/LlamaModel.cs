using System.Runtime.InteropServices;
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

    public LlamaModel(string path, LlamaModelParameters? parameters = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        LlamaBackend.EnsureInitialized();

        var native = (parameters ?? LlamaModelParameters.Default()).ToNative();
        var raw = NativeMethods.llama_model_load_from_file(path, native);
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

        TrainingContextSize = NativeMethods.llama_model_n_ctx_train(raw);
        EmbeddingSize       = NativeMethods.llama_model_n_embd(raw);
        LayerCount          = NativeMethods.llama_model_n_layer(raw);
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
