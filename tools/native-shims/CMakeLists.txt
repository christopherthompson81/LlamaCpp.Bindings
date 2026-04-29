cmake_minimum_required(VERSION 3.18)
project(llamashim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# We don't depend on llama.h or ggml.h at compile time — the shim
# forward-declares the opaque types it touches and pulls in the
# external symbols at link time.
add_library(llamashim SHARED llamashim.cpp)

# Link against the same libllama.so + libggml-base.so the bindings load
# at runtime. LLAMA_NATIVE_DIR points at the runtimes/<rid>/native/
# directory; pass it on the cmake command line:
#   cmake -B build -DLLAMA_NATIVE_DIR=/path/to/runtimes/linux-x64/native
if(NOT DEFINED LLAMA_NATIVE_DIR)
    message(FATAL_ERROR
        "LLAMA_NATIVE_DIR not set. Pass -DLLAMA_NATIVE_DIR=<path> "
        "pointing at the runtimes/<rid>/native directory containing "
        "libllama.so and libggml-base.so.")
endif()

target_link_directories(llamashim PRIVATE ${LLAMA_NATIVE_DIR})
target_link_libraries(llamashim PRIVATE llama ggml-base)

# Use the existing libraries' SONAME-based loading at runtime.
# Embed the native dir as an RPATH so callers don't need to set
# LD_LIBRARY_PATH when libllamashim.so is loaded standalone.
set_target_properties(llamashim PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "${LLAMA_NATIVE_DIR};$ORIGIN"
)

# Strip the `lib` prefix on Windows builds so the .NET LibraryImport
# resolver finds it under the same name "llamashim" on every RID.
if(WIN32)
    set_target_properties(llamashim PROPERTIES PREFIX "")
endif()
