using System.Reflection;
using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native;

/// <summary>
/// Hooks into .NET's P/Invoke resolution to find the llama.cpp native library
/// under <c>runtimes/&lt;rid&gt;/native/</c> relative to the managed DLL.
/// </summary>
/// <remarks>
/// When llama.cpp ships as a NuGet package, the <c>runtimes/</c> layout is
/// resolved automatically by the deps.json graph. For in-tree library projects
/// (our case — the native files are copied from local build output into our
/// own <c>runtimes/</c> folder), there's no deps.json entry for them, so the
/// default resolver never probes there. This resolver closes that gap.
///
/// Registration is idempotent — a static flag ensures we call
/// <c>NativeLibrary.SetDllImportResolver</c> at most once per assembly,
/// regardless of how many times <see cref="LlamaBackend.Initialize"/> is
/// invoked.
/// </remarks>
internal static class NativeLibraryResolver
{
    private static int _registered;

    /// <summary>
    /// Returns the directory where our shipped native libraries live, or null
    /// if it cannot be determined. Used to pass an explicit path to
    /// <c>ggml_backend_load_all_from_path</c> so it finds arch-specific CPU
    /// plugin variants that live alongside libggml.so.
    /// </summary>
    public static string? NativeLibraryDirectory()
    {
        var rid = CurrentRid();
        var assembly = typeof(NativeLibraryResolver).Assembly;

        var baseDir = Path.GetDirectoryName(assembly.Location);
        if (!string.IsNullOrEmpty(baseDir))
        {
            var candidate = Path.Combine(baseDir, "runtimes", rid, "native");
            if (Directory.Exists(candidate)) return candidate;
            if (Directory.Exists(baseDir)) return baseDir; // flattened dev layout
        }

        var appBase = AppContext.BaseDirectory;
        if (!string.IsNullOrEmpty(appBase))
        {
            var candidate = Path.Combine(appBase, "runtimes", rid, "native");
            if (Directory.Exists(candidate)) return candidate;
            if (Directory.Exists(appBase)) return appBase;
        }

        return null;
    }

    /// <summary>Call once at startup (from <c>LlamaBackend.Initialize</c>).</summary>
    public static void Register()
    {
        if (Interlocked.Exchange(ref _registered, 1) != 0) return;
        NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolver).Assembly, Resolve);
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        // Only intercept libraries we ship under runtimes/. Everything else
        // goes to the default resolver unchanged.
        if (libraryName != "llama" && libraryName != "mtmd" && libraryName != "ggml") return IntPtr.Zero;

        var probes = CandidatePaths(assembly, libraryName);
        foreach (var candidate in probes)
        {
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle))
            {
                return handle;
            }
        }

        // Last resort: let the default resolver try. It will fail with a
        // descriptive exception listing the paths it searched if nothing works.
        return IntPtr.Zero;
    }

    private static IEnumerable<string> CandidatePaths(Assembly assembly, string libraryName)
    {
        var rid = CurrentRid();
        var filename = OsLibName(libraryName);

        var baseDir = Path.GetDirectoryName(assembly.Location);
        if (!string.IsNullOrEmpty(baseDir))
        {
            var nativeDir = Path.Combine(baseDir, "runtimes", rid, "native");
            // On Linux, MSBuild dereferences symlinks when copying to the output
            // directory, so libggml.so and libggml.so.0 land as separate regular
            // files with different inodes. libllama.so's NEEDED entry is
            // "libggml.so.0" (the SONAME), so the ELF loader will pick that file.
            // We must load the same file so both paths share one dlopen entry and
            // one backend registry. Probe the SONAME form (.so.N) first.
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var soname = FindSonameFile(nativeDir, libraryName);
                if (soname is not null) yield return soname;
            }
            yield return Path.Combine(nativeDir, filename);
            yield return Path.Combine(baseDir, filename); // flattened dev layout
        }

        // Also try the AppContext base dir — matters when the test host
        // loads the DLL from a different path than assembly.Location.
        var appBase = AppContext.BaseDirectory;
        if (!string.IsNullOrEmpty(appBase))
        {
            var nativeDir = Path.Combine(appBase, "runtimes", rid, "native");
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var soname = FindSonameFile(nativeDir, libraryName);
                if (soname is not null) yield return soname;
            }
            yield return Path.Combine(nativeDir, filename);
            yield return Path.Combine(appBase, filename);
        }
    }

    // Returns the first libX.so.<major> file found in dir (SONAME form — just
    // a single numeric major-version suffix, not the full A.B.C form).
    // Note: Directory.EnumerateFiles("libX.so.*") can match "libX.so" on some
    // .NET versions (treats .* as zero-or-more extension), so we guard with
    // StartsWith and a length check.
    private static string? FindSonameFile(string dir, string libraryName)
    {
        if (!Directory.Exists(dir)) return null;
        var prefix = $"lib{libraryName}.so.";
        return Directory.EnumerateFiles(dir, $"lib{libraryName}.so.*")
            .OrderBy(f => f)
            .FirstOrDefault(f =>
            {
                var name = Path.GetFileName(f);
                if (!name.StartsWith(prefix, StringComparison.Ordinal)) return false;
                var suffix = name[prefix.Length..];
                return suffix.Length > 0 && suffix.All(char.IsAsciiDigit);
            });
    }

    private static string CurrentRid()
    {
        var arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64   => "x64",
            Architecture.Arm64 => "arm64",
            Architecture.X86   => "x86",
            _ => RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(),
        };

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return $"win-{arch}";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))   return $"linux-{arch}";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))     return $"osx-{arch}";

        // Unknown — let the probes fall through and the default resolver handle it.
        return "unknown";
    }

    private static string OsLibName(string libraryName)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return libraryName + ".dll";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))     return "lib" + libraryName + ".dylib";
        return "lib" + libraryName + ".so"; // Linux and fallback
    }
}
