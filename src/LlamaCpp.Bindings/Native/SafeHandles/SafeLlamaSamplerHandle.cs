using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native.SafeHandles;

/// <summary>
/// Owns a <c>llama_sampler *</c> chain. <see cref="ReleaseHandle"/> calls
/// <c>llama_sampler_free</c>, which recursively frees every sub-sampler that
/// was added via <c>llama_sampler_chain_add</c> — individual sub-sampler
/// handles must never be wrapped in their own SafeHandles.
/// </summary>
internal sealed class SafeLlamaSamplerHandle : SafeHandle
{
    public SafeLlamaSamplerHandle() : base(invalidHandleValue: IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr DangerousHandle => handle;

    internal static SafeLlamaSamplerHandle FromUnsafeHandle(IntPtr raw)
    {
        var sh = new SafeLlamaSamplerHandle();
        sh.SetHandle(raw);
        return sh;
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.llama_sampler_free(handle);
            SetHandle(IntPtr.Zero);
        }
        return true;
    }
}
