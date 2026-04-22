using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns a <c>llama_context *</c>. Calls <c>llama_free</c> in ReleaseHandle.
/// Must be disposed strictly before the SafeLlamaModelHandle it was created
/// from (native invariant).
/// </summary>
internal sealed class SafeLlamaContextHandle : SafeHandle
{
    public SafeLlamaContextHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeLlamaContextHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeLlamaContextHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.llama_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
