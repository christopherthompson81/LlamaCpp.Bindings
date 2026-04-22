using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Managed-side mirror of the subset of <c>llama_model_params</c> that callers
/// typically want to set. Defaults match <c>llama_model_default_params()</c>.
/// </summary>
/// <remarks>
/// This is the public surface. The internal struct mirror is kept out of the
/// public API on purpose — its layout is native-version-dependent and we don't
/// want binding consumers to break when we bump llama.cpp.
/// </remarks>
public sealed class LlamaModelParameters
{
    /// <summary>
    /// Number of layers to offload to GPU. <c>-1</c> means "all layers" and
    /// matches <c>llama_model_default_params()</c>. For the 3090 class card
    /// on a 20-35B quant, -1 is the right default. Set to 0 to force CPU.
    /// </summary>
    public int GpuLayerCount { get; set; } = -1;

    /// <summary>Index of the GPU used when <see cref="SplitMode"/> is None.</summary>
    public int MainGpu { get; set; } = 0;

    /// <summary>How to split the model across multiple GPUs.</summary>
    public LlamaSplitMode SplitMode { get; set; } = LlamaSplitMode.Layer;

    /// <summary>Load only vocab metadata, not weights. Rarely useful outside CI.</summary>
    public bool VocabOnly { get; set; } = false;

    /// <summary>Memory-map the GGUF file rather than reading it into RAM.</summary>
    public bool UseMmap { get; set; } = true;

    /// <summary>Lock model memory (requires elevated privileges on Linux).</summary>
    public bool UseMlock { get; set; } = false;

    /// <summary>Validate tensor data during load. Slow but catches corruption.</summary>
    public bool CheckTensors { get; set; } = false;

    internal llama_model_params ToNative()
    {
        var native = NativeMethods.llama_model_default_params();
        native.n_gpu_layers = GpuLayerCount;
        native.main_gpu = MainGpu;
        native.split_mode = (llama_split_mode)SplitMode;
        native.vocab_only = VocabOnly;
        native.use_mmap = UseMmap;
        native.use_mlock = UseMlock;
        native.check_tensors = CheckTensors;
        return native;
    }

    /// <summary>Build a managed snapshot of the native defaults as a starting point.</summary>
    public static LlamaModelParameters Default()
    {
        LlamaBackend.EnsureInitialized();
        var native = NativeMethods.llama_model_default_params();
        return new LlamaModelParameters
        {
            GpuLayerCount = native.n_gpu_layers,
            MainGpu       = native.main_gpu,
            SplitMode     = (LlamaSplitMode)native.split_mode,
            VocabOnly     = native.vocab_only,
            UseMmap       = native.use_mmap,
            UseMlock      = native.use_mlock,
            CheckTensors  = native.check_tensors,
        };
    }
}

public enum LlamaSplitMode
{
    None  = 0,
    Layer = 1,
    Row   = 2,
}
