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
}
