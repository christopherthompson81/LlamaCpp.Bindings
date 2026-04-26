using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// One-shot GGUF quantization driver. Wraps <c>llama_model_quantize</c>:
/// reads a source GGUF, writes a quantized output, and returns when the
/// native call completes.
/// </summary>
/// <remarks>
/// <para>
/// The native call is synchronous and exposes no progress callback.
/// <see cref="QuantizeAsync"/> runs it on a background thread so callers
/// don't have to block. To observe progress, install a log sink via
/// <see cref="LlamaBackend.Initialize"/> — llama.cpp emits one
/// <c>"[ N/M] tensor_name ..."</c> line per tensor through that route.
/// </para>
/// <para>
/// Cancellation is checked before the native call begins; once
/// <c>llama_model_quantize</c> is running, the operation cannot be aborted
/// and will run to completion (the partial output file is then deleted by
/// the binding so callers don't have to clean up after a cancelled
/// pre-flight check).
/// </para>
/// </remarks>
public static class LlamaQuantizer
{
    /// <summary>
    /// Quantize <paramref name="inputPath"/> to <paramref name="outputPath"/>
    /// using <paramref name="parameters"/>. Throws <see cref="LlamaException"/>
    /// if the native call returns a nonzero status.
    /// </summary>
    /// <param name="inputPath">Source GGUF — must exist and be readable.</param>
    /// <param name="outputPath">Destination GGUF — overwritten if it exists.</param>
    /// <param name="parameters">Quantization knobs. Pass <c>null</c> for defaults.</param>
    /// <param name="cancellationToken">
    /// Honored before the native call begins. Mid-flight cancellation is
    /// not supported — see remarks on <see cref="LlamaQuantizer"/>.
    /// </param>
    public static Task QuantizeAsync(
        string inputPath,
        string outputPath,
        LlamaQuantizationParameters? parameters = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(inputPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        LlamaBackend.EnsureInitialized();

        if (!File.Exists(inputPath))
        {
            throw new FileNotFoundException(
                $"Input GGUF not found: {inputPath}", inputPath);
        }

        var p = parameters ?? new LlamaQuantizationParameters();

        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            QuantizeCore(inputPath, outputPath, p);
        }, cancellationToken);
    }

    /// <summary>
    /// Synchronous variant of <see cref="QuantizeAsync"/>. Blocks the calling
    /// thread for the entire duration of the native quantization — typically
    /// seconds for a 1B model, minutes for a 70B. UI callers should prefer
    /// the async overload.
    /// </summary>
    public static void Quantize(
        string inputPath,
        string outputPath,
        LlamaQuantizationParameters? parameters = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(inputPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        LlamaBackend.EnsureInitialized();

        if (!File.Exists(inputPath))
        {
            throw new FileNotFoundException(
                $"Input GGUF not found: {inputPath}", inputPath);
        }

        QuantizeCore(inputPath, outputPath, parameters ?? new LlamaQuantizationParameters());
    }

    private static unsafe void QuantizeCore(
        string inputPath,
        string outputPath,
        LlamaQuantizationParameters parameters)
    {
        var native = parameters.ToNative();

        // Per-tensor type overrides need a pinned native array of
        // llama_model_tensor_override structs (pattern + ggml_type) and
        // the pattern strings must outlive the native call. We allocate
        // both up-front, plant the pointer in native.tt_overrides, and
        // free everything in finally.
        var overrideAllocations = AllocateTensorTypeOverrides(parameters.TensorTypeOverrides, ref native);
        try
        {
            var status = NativeMethods.llama_model_quantize(inputPath, outputPath, &native);
            if (status != 0)
            {
                throw new LlamaException(
                    "llama_model_quantize",
                    (int)status,
                    $"llama_model_quantize returned status {status} (input='{inputPath}', " +
                    $"output='{outputPath}', ftype={parameters.FileType}).");
            }
        }
        finally
        {
            overrideAllocations.Free();
        }
    }

    /// <summary>
    /// Marshal a managed <see cref="LlamaQuantizationParameters.TensorTypeOverrides"/>
    /// list into the native struct's <c>tt_overrides</c> field. The
    /// native side reads an array of <c>llama_model_tensor_override</c>
    /// terminated by an entry whose <c>pattern</c> field is null.
    /// </summary>
    private static OverrideAllocations AllocateTensorTypeOverrides(
        IReadOnlyList<KeyValuePair<string, LlamaTensorType>>? overrides,
        ref llama_model_quantize_params native)
    {
        if (overrides is null || overrides.Count == 0)
        {
            return default;
        }

        // Native struct layout: { const char* pattern; ggml_type type; }
        // = 16 bytes (pointer + 4-byte enum + 4-byte tail pad). Allocate
        // (count + 1) entries — the trailing one has pattern=null and
        // tells the native scanner to stop.
        int entryCount = overrides.Count;
        int byteCount = (entryCount + 1) * llama_model_tensor_override.ExpectedSize;
        IntPtr buffer = Marshal.AllocHGlobal(byteCount);
        var patternPtrs = new IntPtr[entryCount];

        try
        {
            for (int i = 0; i < entryCount; i++)
            {
                var entry = overrides[i];
                if (string.IsNullOrEmpty(entry.Key))
                {
                    throw new ArgumentException(
                        $"TensorTypeOverrides[{i}].Key is empty — would terminate the array early in the native scanner.",
                        nameof(overrides));
                }

                var patternPtr = Marshal.StringToCoTaskMemUTF8(entry.Key);
                patternPtrs[i] = patternPtr;

                int slot = i * llama_model_tensor_override.ExpectedSize;
                Marshal.WriteIntPtr(buffer, slot, patternPtr);
                Marshal.WriteInt32(buffer, slot + IntPtr.Size, (int)entry.Value);
                // 4-byte tail pad after the enum is zero-initialized below.
            }

            // Terminator entry: pattern=null, type=0.
            int tail = entryCount * llama_model_tensor_override.ExpectedSize;
            Marshal.WriteIntPtr(buffer, tail, IntPtr.Zero);
            Marshal.WriteInt32(buffer, tail + IntPtr.Size, 0);

            // Zero the tail-pad bytes for every entry so debug builds of
            // ggml that read past the enum don't see uninitialized memory.
            for (int i = 0; i <= entryCount; i++)
            {
                int padOffset = i * llama_model_tensor_override.ExpectedSize + IntPtr.Size + sizeof(int);
                int padBytes  = llama_model_tensor_override.ExpectedSize - IntPtr.Size - sizeof(int);
                for (int p = 0; p < padBytes; p++) Marshal.WriteByte(buffer, padOffset + p, 0);
            }

            native.tt_overrides = buffer;
            return new OverrideAllocations(buffer, patternPtrs);
        }
        catch
        {
            // Roll back partial allocations if construction fails.
            foreach (var ptr in patternPtrs)
            {
                if (ptr != IntPtr.Zero) Marshal.FreeCoTaskMem(ptr);
            }
            Marshal.FreeHGlobal(buffer);
            throw;
        }
    }

    /// <summary>Disposable holder for the tt_overrides side allocations; <see cref="Free"/> after the native call returns.</summary>
    private readonly struct OverrideAllocations
    {
        private readonly IntPtr _buffer;
        private readonly IntPtr[]? _patternPtrs;

        public OverrideAllocations(IntPtr buffer, IntPtr[] patternPtrs)
        {
            _buffer = buffer;
            _patternPtrs = patternPtrs;
        }

        public void Free()
        {
            if (_patternPtrs is not null)
            {
                foreach (var ptr in _patternPtrs)
                {
                    if (ptr != IntPtr.Zero) Marshal.FreeCoTaskMem(ptr);
                }
            }
            if (_buffer != IntPtr.Zero) Marshal.FreeHGlobal(_buffer);
        }
    }
}
