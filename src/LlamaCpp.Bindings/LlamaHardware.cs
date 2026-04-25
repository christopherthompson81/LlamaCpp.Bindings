using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Compute device discovered by ggml. Provides name + free/total memory
/// so higher layers (auto-configure heuristics, UI diagnostics) can size
/// context and GPU-layer counts against actual hardware. The
/// <see cref="Handle"/> field carries the underlying
/// <c>ggml_backend_dev_t</c> pointer so device-pinning consumers
/// (<see cref="LlamaModelParameters.Devices"/>, override-tensor
/// patterns) can pass it back into native APIs.
/// </summary>
public sealed record LlamaComputeDevice(
    string Name,
    string Description,
    LlamaComputeDeviceType Type,
    long FreeBytes,
    long TotalBytes,
    IntPtr Handle);

/// <summary>
/// Kind of compute device. Mirrors <c>ggml_backend_dev_type</c>.
/// </summary>
public enum LlamaComputeDeviceType
{
    Cpu = 0,
    Gpu = 1,
    /// <summary>Integrated GPU using host memory (Apple Silicon, many laptops).</summary>
    IntegratedGpu = 2,
    /// <summary>Accelerator used alongside a CPU backend (BLAS, AMX).</summary>
    Accelerator = 3,
    /// <summary>Virtual device spanning multiple backing devices.</summary>
    Meta = 4,
}

/// <summary>
/// Process-wide hardware discovery. Enumerates every backend device ggml
/// can see (after <see cref="LlamaBackend.Initialize"/>) and reports
/// per-device memory. Call once on startup or before auto-configure;
/// results are a snapshot, so call again if the user plugs in/out a GPU.
/// </summary>
public static class LlamaHardware
{
    /// <summary>
    /// Enumerate every compute device ggml has registered. Backend plugins
    /// must already be loaded (i.e. <see cref="LlamaBackend.Initialize"/>
    /// has run) — otherwise only the CPU device shows up.
    /// </summary>
    public static unsafe IReadOnlyList<LlamaComputeDevice> EnumerateDevices()
    {
        LlamaBackend.EnsureInitialized();

        var count = (int)NativeMethods.ggml_backend_dev_count();
        if (count <= 0) return Array.Empty<LlamaComputeDevice>();

        var result = new List<LlamaComputeDevice>(count);
        for (nuint i = 0; i < (nuint)count; i++)
        {
            var devPtr = NativeMethods.ggml_backend_dev_get(i);
            if (devPtr == IntPtr.Zero) continue;

            var namePtr = NativeMethods.ggml_backend_dev_name(devPtr);
            var descPtr = NativeMethods.ggml_backend_dev_description(devPtr);
            var type    = NativeMethods.ggml_backend_dev_type(devPtr);

            nuint freeB = 0, totalB = 0;
            NativeMethods.ggml_backend_dev_memory(devPtr, &freeB, &totalB);

            result.Add(new LlamaComputeDevice(
                Name: Marshal.PtrToStringUTF8(namePtr) ?? "(unknown)",
                Description: Marshal.PtrToStringUTF8(descPtr) ?? string.Empty,
                Type: (LlamaComputeDeviceType)type,
                FreeBytes: (long)freeB,
                TotalBytes: (long)totalB,
                Handle: devPtr));
        }
        return result;
    }

    /// <summary>
    /// Look up a single device by its ggml-reported name (e.g. <c>"CUDA0"</c>,
    /// <c>"CPU"</c>). Case-insensitive. Returns <c>null</c> when no device
    /// matches — operators get a clearer error than a confusing native
    /// failure deep in load.
    /// </summary>
    public static LlamaComputeDevice? FindDeviceByName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        foreach (var d in EnumerateDevices())
        {
            if (string.Equals(d.Name, name, StringComparison.OrdinalIgnoreCase))
            {
                return d;
            }
        }
        return null;
    }

    /// <summary>
    /// Sum of <c>FreeBytes</c> across every GPU (dedicated or integrated)
    /// device. Zero on CPU-only systems or when backend plugins failed to
    /// load a GPU backend.
    /// </summary>
    public static long GetTotalFreeVramBytes()
    {
        long total = 0;
        foreach (var d in EnumerateDevices())
        {
            if (d.Type is LlamaComputeDeviceType.Gpu or LlamaComputeDeviceType.IntegratedGpu)
            {
                total += d.FreeBytes;
            }
        }
        return total;
    }
}
