using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns a <c>llama_model *</c>. Calls <c>llama_model_free</c> in ReleaseHandle.
/// Public API never sees the raw pointer; everything flows through the SafeHandle.
/// </summary>
internal sealed class SafeLlamaModelHandle : SafeHandle
{
    public SafeLlamaModelHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeLlamaModelHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeLlamaModelHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.llama_model_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
