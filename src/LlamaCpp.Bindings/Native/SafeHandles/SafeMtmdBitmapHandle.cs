using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns an <c>mtmd_bitmap *</c>. Calls <c>mtmd_bitmap_free</c> in ReleaseHandle.
/// Bitmaps are independent of the mtmd context once constructed — they only
/// need the context during <c>mtmd_helper_bitmap_init_from_*</c> to pick the
/// right preprocessing pipeline.
/// </summary>
internal sealed class SafeMtmdBitmapHandle : SafeHandle
{
    public SafeMtmdBitmapHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeMtmdBitmapHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeMtmdBitmapHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.mtmd_bitmap_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
