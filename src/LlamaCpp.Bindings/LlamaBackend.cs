using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// One-time process-wide initialisation for llama.cpp. Call
/// <see cref="Initialize"/> once at application startup before touching any
/// other <c>Llama*</c> type. Threadsafe and idempotent.
/// </summary>
/// <remarks>
/// Also runs the struct layout assertions — if the native binary's struct sizes
/// have drifted from the pinned header, we throw here rather than let a later
/// P/Invoke silently corrupt memory.
/// </remarks>
public static class LlamaBackend
{
    private static readonly object _gate = new();
    private static bool _initialized;

    // Kept alive to prevent the delegate from being GC'd while native code holds a pointer to it.
    private static NativeMethods.GgmlLogCallback? _logCallback;
    private static Action<LlamaLogLevel, string>? _logSink;

    /// <summary>
    /// Initialize llama.cpp. Runs struct layout checks, calls
    /// <c>llama_backend_init</c>, and optionally routes native logs to
    /// <paramref name="logSink"/>.
    /// </summary>
    /// <param name="logSink">
    /// Invoked for every native log line. If null, native logs go to stderr
    /// (llama.cpp's default). The sink is called from native threads — do not
    /// touch UI state from it without marshalling.
    /// </param>
    public static void Initialize(Action<LlamaLogLevel, string>? logSink = null)
    {
        lock (_gate)
        {
            if (_initialized)
            {
                // Allow re-binding the log sink even after init.
                if (logSink is not null)
                {
                    InstallLogSink(logSink);
                }
                return;
            }

            // Teach the P/Invoke resolver where to find libllama.so before the
            // first NativeMethods call. Safe to call before Verify() because
            // Verify() is pure managed.
            NativeLibraryResolver.Register();

            NativeLayout.Verify();

            if (logSink is not null)
            {
                InstallLogSink(logSink);
            }

            NativeMethods.llama_backend_init();
            _initialized = true;
        }
    }

    /// <summary>
    /// Tears down llama.cpp. Rarely needed — llama.cpp itself only uses this
    /// for MPI — but exposed for completeness and for test-harness symmetry.
    /// After this call, <see cref="Initialize"/> must be called again before
    /// any other Llama* type is used.
    /// </summary>
    public static void Shutdown()
    {
        lock (_gate)
        {
            if (!_initialized) return;

            // Detach our log callback before the backend goes away.
            NativeMethods.llama_log_set(IntPtr.Zero, IntPtr.Zero);
            _logCallback = null;
            _logSink = null;

            NativeMethods.llama_backend_free();
            _initialized = false;
        }
    }

    internal static void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException(
                "LlamaBackend.Initialize() must be called before any Llama* type is constructed.");
        }
    }

    private static void InstallLogSink(Action<LlamaLogLevel, string> sink)
    {
        _logSink = sink;
        _logCallback = LogTrampoline;
        var fp = Marshal.GetFunctionPointerForDelegate(_logCallback);
        NativeMethods.llama_log_set(fp, IntPtr.Zero);
    }

    private static void LogTrampoline(ggml_log_level level, IntPtr text, IntPtr userData)
    {
        _ = userData;
        var sink = _logSink;
        if (sink is null || text == IntPtr.Zero) return;

        var message = Marshal.PtrToStringUTF8(text);
        if (string.IsNullOrEmpty(message)) return;

        // llama.cpp's native log lines come in already-formatted with trailing
        // newlines. Trim so hosts don't have to.
        sink(MapLevel(level), message.TrimEnd('\r', '\n'));
    }

    private static LlamaLogLevel MapLevel(ggml_log_level level) => level switch
    {
        ggml_log_level.GGML_LOG_LEVEL_DEBUG => LlamaLogLevel.Debug,
        ggml_log_level.GGML_LOG_LEVEL_INFO  => LlamaLogLevel.Info,
        ggml_log_level.GGML_LOG_LEVEL_WARN  => LlamaLogLevel.Warn,
        ggml_log_level.GGML_LOG_LEVEL_ERROR => LlamaLogLevel.Error,
        ggml_log_level.GGML_LOG_LEVEL_CONT  => LlamaLogLevel.Continuation,
        _ => LlamaLogLevel.Info,
    };

    /// <summary>Capability snapshot — useful for the hardware heuristics.</summary>
    public static bool SupportsGpuOffload()
    {
        EnsureInitialized();
        return NativeMethods.llama_supports_gpu_offload();
    }

    public static bool SupportsMmap()
    {
        EnsureInitialized();
        return NativeMethods.llama_supports_mmap();
    }

    public static bool SupportsMlock()
    {
        EnsureInitialized();
        return NativeMethods.llama_supports_mlock();
    }

    public static int MaxDevices()
    {
        EnsureInitialized();
        return (int)NativeMethods.llama_max_devices();
    }

    /// <summary>Max sequences the native build will allow in a single batch.</summary>
    public static int MaxParallelSequences()
    {
        EnsureInitialized();
        return (int)NativeMethods.llama_max_parallel_sequences();
    }

    /// <summary>True if this native build was compiled with RPC support.</summary>
    public static bool SupportsRpc()
    {
        EnsureInitialized();
        return NativeMethods.llama_supports_rpc();
    }

    /// <summary>
    /// Backend identification string (e.g. <c>"AVX = 1 | AVX2 = 1 | ... | CUDA = 1"</c>).
    /// Useful in bug reports and "what does this machine support" UIs.
    /// </summary>
    public static string SystemInfo()
    {
        EnsureInitialized();
        var ptr = NativeMethods.llama_print_system_info();
        return ptr == IntPtr.Zero
            ? string.Empty
            : System.Runtime.InteropServices.Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
    }

    /// <summary>
    /// Initialise NUMA support. Call once on a NUMA system before loading
    /// models. Has no effect on non-NUMA systems.
    /// </summary>
    public static void InitializeNuma(LlamaNumaStrategy strategy)
    {
        EnsureInitialized();
        NativeMethods.llama_numa_init((ggml_numa_strategy)strategy);
    }
}

/// <summary>
/// NUMA memory placement strategy for CPU inference on multi-socket systems.
/// </summary>
public enum LlamaNumaStrategy
{
    Disabled   = 0,
    /// <summary>Spread allocations across NUMA nodes.</summary>
    Distribute = 1,
    /// <summary>Pin each worker to a single node.</summary>
    Isolate    = 2,
    /// <summary>Let the system NUMA policy decide (honour <c>numactl</c> if set).</summary>
    Numactl    = 3,
    /// <summary>Duplicate model weights on every NUMA node — more memory, less cross-node traffic.</summary>
    Mirror     = 4,
}

/// <summary>Log level passed to a <see cref="LlamaBackend.Initialize"/> sink.</summary>
public enum LlamaLogLevel
{
    Debug,
    Info,
    Warn,
    Error,
    /// <summary>Continuation of the previous line — llama.cpp sometimes splits a single message across multiple callback invocations.</summary>
    Continuation,
}
