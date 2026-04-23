using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns an <c>mtmd_context *</c>. Calls <c>mtmd_free</c> in ReleaseHandle.
/// Must be disposed strictly before the SafeLlamaModelHandle it was created
/// from, and before any <see cref="SafeMtmdBitmapHandle"/> or
/// <see cref="SafeMtmdInputChunksHandle"/> allocated against it.
/// </summary>
internal sealed class SafeMtmdContextHandle : SafeHandle
{
    public SafeMtmdContextHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeMtmdContextHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeMtmdContextHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.mtmd_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
