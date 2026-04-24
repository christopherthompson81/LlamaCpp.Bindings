using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns a <c>llama_adapter_lora *</c>. <see cref="ReleaseHandle"/> calls
/// <c>llama_adapter_lora_free</c>. The native invariant is that an adapter
/// must outlive any context it's attached to and must be freed strictly
/// before its associated model. The pinned llama.cpp will free any
/// still-alive adapters when the model is freed, but our wrapper performs
/// an explicit, deterministic free so callers can control the ordering.
/// </summary>
internal sealed class SafeLlamaAdapterHandle : SafeHandle
{
    public SafeLlamaAdapterHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeLlamaAdapterHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeLlamaAdapterHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.llama_adapter_lora_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
