using System.Runtime.InteropServices;
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

    /// <summary>
    /// Use direct I/O (<c>O_DIRECT</c>) when reading the GGUF — bypasses the
    /// kernel page cache. Useful for cold loads on machines whose page cache
    /// is needed for other workloads, but pays the cost of slower mmap.
    /// </summary>
    public bool UseDirectIo { get; set; } = false;

    /// <summary>
    /// When false (default), llama.cpp may pin model tensors into host
    /// memory accessible from the GPU. <c>true</c> forces it to skip the
    /// host-pinning step — saves shared memory at the cost of slower
    /// CPU↔GPU transfers.
    /// </summary>
    public bool NoHost { get; set; } = false;

    /// <summary>
    /// Allow llama.cpp to use "extra buffer types" that some backends provide
    /// for tensor repacking (e.g., AMX-friendly layouts on supported CPUs).
    /// llama-server's <c>--repack</c> flag corresponds to this. Default
    /// matches <c>llama_model_default_params()</c>.
    /// </summary>
    public bool UseExtraBufts { get; set; } = true;

    /// <summary>
    /// Restrict the model load to a specific set of compute devices (in
    /// order). When non-null and non-empty, llama.cpp ignores its built-in
    /// device discovery and uses only these devices. <c>null</c> or empty
    /// = use every registered device (default).
    /// </summary>
    /// <remarks>
    /// Devices come from <see cref="LlamaHardware.EnumerateDevices"/> /
    /// <see cref="LlamaHardware.FindDeviceByName"/>. Their handles must
    /// outlive the load call only — once <see cref="LlamaModel"/>'s ctor
    /// returns, the array is freed.
    /// </remarks>
    public IReadOnlyList<LlamaComputeDevice>? Devices { get; set; }

    /// <summary>
    /// Per-device proportion of the model to offload (length must be
    /// <c>llama_max_devices()</c> when non-null; the binding pads shorter
    /// inputs with zeros and rejects longer ones). Pairs with
    /// <see cref="LlamaSplitMode.Layer"/> / <see cref="LlamaSplitMode.Row"/>
    /// to control which device gets which fraction.
    /// </summary>
    public IReadOnlyList<float>? TensorSplit { get; set; }

    /// <summary>
    /// Pattern-driven tensor placement overrides. Each entry routes
    /// tensors whose names match <see cref="LlamaTensorBuftOverride.Pattern"/>
    /// to <see cref="LlamaTensorBuftOverride.BufferType"/> instead of
    /// their default device. Used by llama-server's
    /// <c>--override-tensor</c> and <c>--cpu-moe</c> presets.
    /// </summary>
    /// <remarks>
    /// The list is capped at <c>llama_max_tensor_buft_overrides()</c>
    /// (currently 256). Patterns are POSIX-style regex; matches are
    /// evaluated in list order and the first one wins.
    /// </remarks>
    public IReadOnlyList<LlamaTensorBuftOverride>? TensorBuftOverrides { get; set; }

    internal llama_model_params ToNative()
    {
        var native = NativeMethods.llama_model_default_params();
        native.n_gpu_layers   = GpuLayerCount;
        native.main_gpu       = MainGpu;
        native.split_mode     = (llama_split_mode)SplitMode;
        native.vocab_only     = VocabOnly;
        native.use_mmap       = UseMmap;
        native.use_mlock      = UseMlock;
        native.check_tensors  = CheckTensors;
        native.use_direct_io  = UseDirectIo;
        native.no_host        = NoHost;
        native.use_extra_bufts = UseExtraBufts;
        return native;
    }

    /// <summary>
    /// Build the native struct with side allocations (devices + tensor_split
    /// pointers) pinned for the lifetime of the returned handle. Caller
    /// must dispose the handle <em>after</em> the native consumer
    /// (e.g. <c>llama_model_load_from_file</c>) has returned.
    /// </summary>
    internal PinnedNative Pin()
    {
        var native = ToNative();
        IntPtr devicesBuf = IntPtr.Zero;
        IntPtr tensorSplitBuf = IntPtr.Zero;
        IntPtr overridesBuf = IntPtr.Zero;
        var overridePatternBufs = new List<IntPtr>();

        try
        {
            if (Devices is { Count: > 0 } devs)
            {
                // NULL-terminated ggml_backend_dev_t array. One extra slot
                // for the terminator; llama.cpp scans until it sees NULL.
                devicesBuf = Marshal.AllocHGlobal((devs.Count + 1) * IntPtr.Size);
                for (int i = 0; i < devs.Count; i++)
                {
                    Marshal.WriteIntPtr(devicesBuf, i * IntPtr.Size, devs[i].Handle);
                }
                Marshal.WriteIntPtr(devicesBuf, devs.Count * IntPtr.Size, IntPtr.Zero);
                native.devices = devicesBuf;
            }

            if (TensorSplit is { Count: > 0 } split)
            {
                int max = (int)NativeMethods.llama_max_devices();
                if (split.Count > max)
                {
                    throw new ArgumentException(
                        $"TensorSplit has {split.Count} entries but llama_max_devices() reports {max}.",
                        nameof(TensorSplit));
                }
                // Always allocate the full max-devices block; llama.cpp
                // reads up to max regardless of what we supply.
                int byteCount = max * sizeof(float);
                tensorSplitBuf = Marshal.AllocHGlobal(byteCount);
                unsafe
                {
                    var dst = (float*)tensorSplitBuf;
                    for (int i = 0; i < max; i++)
                    {
                        dst[i] = i < split.Count ? split[i] : 0f;
                    }
                }
                native.tensor_split = tensorSplitBuf;
            }

            if (TensorBuftOverrides is { Count: > 0 } overrides)
            {
                int max = (int)NativeMethods.llama_max_tensor_buft_overrides();
                if (overrides.Count > max)
                {
                    throw new ArgumentException(
                        $"TensorBuftOverrides has {overrides.Count} entries but " +
                        $"llama_max_tensor_buft_overrides() reports {max}.",
                        nameof(TensorBuftOverrides));
                }

                // Each entry is { const char* pattern; void* buft } — two
                // pointer-sized fields. The list is NULL-terminated by an
                // entry whose pattern field is null.
                int entrySize = IntPtr.Size * 2;
                int byteCount = (overrides.Count + 1) * entrySize;
                overridesBuf = Marshal.AllocHGlobal(byteCount);

                for (int i = 0; i < overrides.Count; i++)
                {
                    var entry = overrides[i];
                    if (entry is null)
                    {
                        throw new ArgumentException(
                            $"TensorBuftOverrides[{i}] is null.", nameof(TensorBuftOverrides));
                    }
                    if (string.IsNullOrEmpty(entry.Pattern))
                    {
                        throw new ArgumentException(
                            $"TensorBuftOverrides[{i}].Pattern is empty — would terminate the " +
                            "list early in the native struct layout.", nameof(TensorBuftOverrides));
                    }

                    // Allocate UTF-8 bytes for the pattern. Patterns are
                    // typically ASCII regex but we use UTF-8 for
                    // consistency with the rest of the binding's string
                    // marshalling. PinnedNative.Dispose frees these.
                    var patternPtr = Marshal.StringToCoTaskMemUTF8(entry.Pattern);
                    overridePatternBufs.Add(patternPtr);

                    int slot = i * entrySize;
                    Marshal.WriteIntPtr(overridesBuf, slot, patternPtr);
                    Marshal.WriteIntPtr(overridesBuf, slot + IntPtr.Size, entry.BufferType.Handle);
                }
                // Terminator
                int tail = overrides.Count * entrySize;
                Marshal.WriteIntPtr(overridesBuf, tail, IntPtr.Zero);
                Marshal.WriteIntPtr(overridesBuf, tail + IntPtr.Size, IntPtr.Zero);

                native.tensor_buft_overrides = overridesBuf;
            }
        }
        catch
        {
            if (devicesBuf != IntPtr.Zero) Marshal.FreeHGlobal(devicesBuf);
            if (tensorSplitBuf != IntPtr.Zero) Marshal.FreeHGlobal(tensorSplitBuf);
            if (overridesBuf != IntPtr.Zero) Marshal.FreeHGlobal(overridesBuf);
            foreach (var p in overridePatternBufs) Marshal.FreeHGlobal(p);
            throw;
        }

        return new PinnedNative(native, devicesBuf, tensorSplitBuf, overridesBuf, overridePatternBufs);
    }

    /// <summary>
    /// Disposable holder for a native <c>llama_model_params</c> together
    /// with any unmanaged side allocations the params reference. Dispose
    /// frees the side allocations; the native struct itself is a value type
    /// and goes away with the consumer.
    /// </summary>
    internal sealed class PinnedNative : IDisposable
    {
        public llama_model_params Native;
        private IntPtr _devicesBuf;
        private IntPtr _tensorSplitBuf;
        private IntPtr _overridesBuf;
        private List<IntPtr>? _overridePatternBufs;
        private bool _disposed;

        public PinnedNative(
            llama_model_params native,
            IntPtr devicesBuf,
            IntPtr tensorSplitBuf,
            IntPtr overridesBuf,
            List<IntPtr>? overridePatternBufs)
        {
            Native = native;
            _devicesBuf = devicesBuf;
            _tensorSplitBuf = tensorSplitBuf;
            _overridesBuf = overridesBuf;
            _overridePatternBufs = overridePatternBufs;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_devicesBuf != IntPtr.Zero) Marshal.FreeHGlobal(_devicesBuf);
            if (_tensorSplitBuf != IntPtr.Zero) Marshal.FreeHGlobal(_tensorSplitBuf);
            if (_overridesBuf != IntPtr.Zero) Marshal.FreeHGlobal(_overridesBuf);
            if (_overridePatternBufs is not null)
            {
                foreach (var p in _overridePatternBufs)
                {
                    Marshal.FreeCoTaskMem(p);
                }
                _overridePatternBufs = null;
            }
            _devicesBuf = IntPtr.Zero;
            _tensorSplitBuf = IntPtr.Zero;
            _overridesBuf = IntPtr.Zero;
        }
    }

    /// <summary>Build a managed snapshot of the native defaults as a starting point.</summary>
    public static LlamaModelParameters Default()
    {
        LlamaBackend.EnsureInitialized();
        var native = NativeMethods.llama_model_default_params();
        return new LlamaModelParameters
        {
            GpuLayerCount  = native.n_gpu_layers,
            MainGpu        = native.main_gpu,
            SplitMode      = (LlamaSplitMode)native.split_mode,
            VocabOnly      = native.vocab_only,
            UseMmap        = native.use_mmap,
            UseMlock       = native.use_mlock,
            CheckTensors   = native.check_tensors,
            UseDirectIo    = native.use_direct_io,
            NoHost         = native.no_host,
            UseExtraBufts  = native.use_extra_bufts,
        };
    }
}

public enum LlamaSplitMode
{
    None  = 0,
    Layer = 1,
    Row   = 2,
}
