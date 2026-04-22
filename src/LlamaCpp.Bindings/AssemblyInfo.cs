using System.Runtime.CompilerServices;

// Required for LibraryImport with structs passed by value. We do bit-exact
// struct mirroring ourselves ([StructLayout(Sequential)] + manual [MarshalAs]
// on each bool) and don't want the runtime second-guessing any field's
// representation — it would both slow the P/Invoke and hide layout drift.
[assembly: System.Runtime.CompilerServices.DisableRuntimeMarshalling]

// Struct-layout and internal-API tests need to see internal types directly.
// This is test-only — no public API is leaked through it.
[assembly: InternalsVisibleTo("LlamaCpp.Bindings.Tests")]
