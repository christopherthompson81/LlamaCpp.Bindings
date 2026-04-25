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

/// <summary>
/// A ggml backend buffer type — the destination for a model tensor. Each
/// device has a primary buft (used by default) and may have a host-pinned
/// buft for transfers between CPU and that device. Used as the target of
/// <see cref="LlamaTensorBuftOverride"/>.
/// </summary>
/// <remarks>
/// Lifetime is owned by ggml; the underlying pointer lives as long as the
/// backend itself. There is no Dispose — wrappers are cheap value records.
/// </remarks>
public sealed record LlamaBufferType(IntPtr Handle, string Name)
{
    /// <summary>
    /// Primary buffer type for <paramref name="device"/> — where tensors
    /// stored on that device live by default.
    /// </summary>
    public static LlamaBufferType From(LlamaComputeDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var buft = NativeMethods.ggml_backend_dev_buffer_type(device.Handle);
        if (buft == IntPtr.Zero)
        {
            throw new LlamaException(
                nameof(NativeMethods.ggml_backend_dev_buffer_type),
                $"Device '{device.Name}' has no primary buffer type.");
        }
        return Wrap(buft);
    }

    /// <summary>
    /// Host-pinned buffer type for <paramref name="device"/> — used for
    /// CPU↔device transfers (e.g. CUDA's pinned-host allocator). Returns
    /// <c>null</c> when the device has no host buft.
    /// </summary>
    public static LlamaBufferType? HostFrom(LlamaComputeDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var buft = NativeMethods.ggml_backend_dev_host_buffer_type(device.Handle);
        return buft == IntPtr.Zero ? null : Wrap(buft);
    }

    private static LlamaBufferType Wrap(IntPtr buft)
    {
        var namePtr = NativeMethods.ggml_backend_buft_name(buft);
        var name = Marshal.PtrToStringUTF8(namePtr) ?? "(unknown)";
        return new LlamaBufferType(buft, name);
    }
}

/// <summary>
/// Pattern → buffer-type override applied during model load. Tensors whose
/// names match the regex pattern get loaded into
/// <see cref="BufferType"/> instead of their default location. llama-server's
/// <c>--override-tensor</c> and <c>--cpu-moe</c> are both built on this.
/// </summary>
/// <remarks>
/// Patterns are POSIX-style regexes per llama.cpp's tensor matcher
/// (e.g. <c>"\.ffn_(up|down|gate)_exps"</c>). Multiple overrides are
/// matched in order; the first match wins. The pattern string is not
/// retained by llama.cpp past load — we copy it into unmanaged memory
/// for the load call's duration via <see cref="LlamaModelParameters.Pin"/>.
/// </remarks>
public sealed record LlamaTensorBuftOverride(string Pattern, LlamaBufferType BufferType);
