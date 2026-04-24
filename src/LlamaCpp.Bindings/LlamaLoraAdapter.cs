using System.Runtime.InteropServices;
using System.Text;
using LlamaCpp.Bindings.Native;
using LlamaCpp.Bindings.Native.SafeHandles;

namespace LlamaCpp.Bindings;

/// <summary>
/// A loaded LoRA adapter, bound to a specific base <see cref="LlamaModel"/>.
/// Adapters are tensor deltas on top of the base model; attaching one to a
/// <see cref="LlamaContext"/> (via <see cref="LlamaContext.AttachLoraAdapter"/>)
/// blends those deltas into inference with a caller-supplied scale.
/// </summary>
/// <remarks>
/// <para>
/// Lifetime: the adapter must outlive every context it is attached to, and
/// must be disposed strictly before its base <see cref="LlamaModel"/>.
/// </para>
/// <para>
/// Adapters are not transferable across base models. The tensor shapes encode
/// the base's layer/head layout — attaching a Qwen LoRA to a Llama model
/// will fail at attach time.
/// </para>
/// </remarks>
public sealed class LlamaLoraAdapter : IDisposable
{
    private readonly SafeLlamaAdapterHandle _handle;
    private readonly LlamaModel _baseModel;
    private IReadOnlyDictionary<string, string>? _metadata;
    private int[]? _aloraInvocationTokens;
    private bool _aloraProbed;
    private bool _disposed;

    /// <summary>Path the adapter was loaded from.</summary>
    public string AdapterPath { get; }

    /// <summary>
    /// The base model this adapter is bound to. An adapter loaded against
    /// one model cannot be attached to a context belonging to a different
    /// model.
    /// </summary>
    public LlamaModel BaseModel
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _baseModel;
        }
    }

    /// <summary>
    /// Adapter GGUF metadata (key/value pairs). Materialised on first access
    /// and cached. Typical entries include the adapter's name, rank (alpha),
    /// and training notes.
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _metadata ??= LoadMetadata();
        }
    }

    /// <summary>
    /// For activated-LoRA (alora) adapters, the token sequence that triggers
    /// activation mid-prompt. Null for standard LoRA adapters. The array is a
    /// copy — the caller may mutate it freely.
    /// </summary>
    public IReadOnlyList<int>? AloraInvocationTokens
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ProbeAlora();
            return _aloraInvocationTokens;
        }
    }

    /// <summary>
    /// Load a LoRA adapter from a GGUF file, binding it to
    /// <paramref name="baseModel"/>.
    /// </summary>
    /// <exception cref="FileNotFoundException">File does not exist.</exception>
    /// <exception cref="LlamaException">
    /// Native load failed — common causes: adapter shape incompatible with
    /// <paramref name="baseModel"/>, corrupt GGUF, or unsupported quant.
    /// </exception>
    public static LlamaLoraAdapter LoadFromFile(LlamaModel baseModel, string path)
    {
        ArgumentNullException.ThrowIfNull(baseModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"LoRA adapter file not found: {path}", path);

        LlamaBackend.EnsureInitialized();

        var raw = NativeMethods.llama_adapter_lora_init(baseModel.Handle.DangerousHandle, path);
        if (raw == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_adapter_lora_init),
                $"Failed to load LoRA adapter from '{path}'. Common causes: adapter shape " +
                $"incompatible with the base model (different architecture, layer count, or " +
                $"head count), corrupt GGUF, or unsupported quantization. Check the native log.");
        }
        return new LlamaLoraAdapter(baseModel, path, raw);
    }

    private LlamaLoraAdapter(LlamaModel baseModel, string path, IntPtr raw)
    {
        _baseModel = baseModel;
        AdapterPath = path;
        _handle = SafeLlamaAdapterHandle.FromUnsafeHandle(raw);
    }

    internal SafeLlamaAdapterHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    private unsafe IReadOnlyDictionary<string, string> LoadMetadata()
    {
        var raw = _handle.DangerousHandle;
        var count = NativeMethods.llama_adapter_meta_count(raw);
        if (count <= 0) return new Dictionary<string, string>(0);

        var dict = new Dictionary<string, string>(count, StringComparer.Ordinal);
        for (int i = 0; i < count; i++)
        {
            int idx = i;
            var key = ProbeUtf8(
                (byte* buf, nuint sz) => NativeMethods.llama_adapter_meta_key_by_index(raw, idx, buf, sz));
            if (string.IsNullOrEmpty(key)) continue;
            var val = ProbeUtf8(
                (byte* buf, nuint sz) => NativeMethods.llama_adapter_meta_val_str_by_index(raw, idx, buf, sz))
                ?? string.Empty;
            dict[key] = val;
        }
        return dict;
    }

    private unsafe void ProbeAlora()
    {
        if (_aloraProbed) return;
        _aloraProbed = true;

        var raw = _handle.DangerousHandle;
        var count = NativeMethods.llama_adapter_get_alora_n_invocation_tokens(raw);
        if (count == 0)
        {
            _aloraInvocationTokens = null;
            return;
        }

        var ptr = NativeMethods.llama_adapter_get_alora_invocation_tokens(raw);
        if (ptr == null)
        {
            _aloraInvocationTokens = null;
            return;
        }

        var buf = new int[checked((int)count)];
        new ReadOnlySpan<int>(ptr, buf.Length).CopyTo(buf);
        _aloraInvocationTokens = buf;
    }

    /// <summary>
    /// Look up a single adapter metadata value by key. More efficient than
    /// enumerating <see cref="Metadata"/> when only one entry is needed.
    /// </summary>
    public unsafe string? GetMetadata(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var raw = _handle.DangerousHandle;
        return ProbeUtf8(
            (byte* buf, nuint sz) => NativeMethods.llama_adapter_meta_val_str(raw, key, buf, sz));
    }

    private unsafe delegate int ProbeCall(byte* buf, nuint size);

    private static unsafe string? ProbeUtf8(ProbeCall call)
    {
        const int StackBuf = 256;
        byte* stackPtr = stackalloc byte[StackBuf];
        int needed = call(stackPtr, StackBuf);
        if (needed < 0) return null;

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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
