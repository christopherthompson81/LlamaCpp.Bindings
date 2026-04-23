using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns an <c>mtmd_input_chunks *</c>. Calls <c>mtmd_input_chunks_free</c>
/// in ReleaseHandle, which transitively frees every chunk (and their image
/// token buffers) inside.
/// </summary>
internal sealed class SafeMtmdInputChunksHandle : SafeHandle
{
    public SafeMtmdInputChunksHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeMtmdInputChunksHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeMtmdInputChunksHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.mtmd_input_chunks_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
