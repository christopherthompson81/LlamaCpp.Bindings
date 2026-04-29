// llamashim — thin C shim exposing llama.cpp internals that the public
// llama.h doesn't surface but that the bindings need for in-place tensor
// data updates (the AdaptiveQuantization profile-builder hot path).
//
// The internal symbol we need:
//
//   const std::vector<std::pair<std::string, ggml_tensor*>> &
//       llama_internal_get_tensor_map(const llama_model * model);
//
// is defined in llama.cpp's src/llama-model.cpp and exported from
// libllama.so. It returns the model's tensors_by_name vector — the
// flat (name, ggml_tensor*) list assembled at load time.
//
// This shim re-exports two stable C entry points:
//
//   llamashim_get_model_tensor(model, name)
//       Linear scan of the tensor map by name. Returns ggml_tensor*
//       on hit, NULL on miss.
//
//   llamashim_set_tensor_data(tensor, data, offset, size)
//       Calls ggml_backend_tensor_set, which dispatches to the
//       backend-specific writer (CUDA D2H, CPU memcpy, etc.) based on
//       which backend owns the tensor's buffer.
//
// Maintenance: this shim is tightly coupled to libllama's internal
// vector layout. Re-validate on every llama.cpp pin bump — the
// llama_internal_get_tensor_map signature has been stable since at
// least b6500 but is not part of the documented C API.

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Forward-declare the llama.cpp opaque types so we don't pull in the
// full headers — keeps the shim's compile dependencies minimal.
struct llama_model;
struct ggml_tensor;

// The internal C++ symbol we're reaching into. Declared with the
// libstdc++ ABI it was compiled with — link-time will pick the right
// mangled name from libllama.so.
extern const std::vector<std::pair<std::string, ggml_tensor *>> &
llama_internal_get_tensor_map(const llama_model * model);

// ggml_backend_tensor_set / ggml_backend_tensor_get are part of
// libggml-base's stable C API.
extern "C" void ggml_backend_tensor_set(
    ggml_tensor * tensor, const void * data, size_t offset, size_t size);
extern "C" void ggml_backend_tensor_get(
    const ggml_tensor * tensor, void * data, size_t offset, size_t size);

extern "C" {

// Returns the ggml_tensor* matching `name` (exact match, case-sensitive)
// or NULL if no such tensor exists in the model.
ggml_tensor * llamashim_get_model_tensor(const llama_model * model, const char * name) {
    if (model == nullptr || name == nullptr) {
        return nullptr;
    }
    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & entry : tensors) {
        if (entry.first == name) {
            return entry.second;
        }
    }
    return nullptr;
}

// Writes `size` bytes from `data` into `tensor`'s buffer at `offset`.
// Dispatches to the tensor's owning backend (CUDA, CPU, Metal, …).
// Returns 0 on success, -1 if `tensor` is NULL.
int llamashim_set_tensor_data(ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (tensor == nullptr) {
        return -1;
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return 0;
}

// Reads `size` bytes from `tensor`'s buffer at `offset` into `data`.
// Symmetric companion to llamashim_set_tensor_data; primarily for
// round-trip "did the write take effect?" verification — if the
// bytes we just wrote come back identical, the surgery is real.
int llamashim_get_tensor_data(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (tensor == nullptr) {
        return -1;
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
    return 0;
}

// Returns the number of tensors in the model. Useful for iteration /
// pre-flight sizing on the binding side (e.g., bulk-restore caches).
size_t llamashim_get_tensor_count(const llama_model * model) {
    if (model == nullptr) {
        return 0;
    }
    return llama_internal_get_tensor_map(model).size();
}

// Returns a pointer to the i-th (name, tensor) pair's name, or NULL if
// out of range. The pointer is valid for the lifetime of the model.
// Provided so callers can enumerate names without C++ ABI exposure.
const char * llamashim_get_tensor_name_at(const llama_model * model, size_t index) {
    if (model == nullptr) {
        return nullptr;
    }
    const auto & tensors = llama_internal_get_tensor_map(model);
    if (index >= tensors.size()) {
        return nullptr;
    }
    return tensors[index].first.c_str();
}

// Returns the ggml_tensor* at index `i`, or NULL if out of range.
// Pairs with llamashim_get_tensor_name_at for index-based iteration.
ggml_tensor * llamashim_get_tensor_at(const llama_model * model, size_t index) {
    if (model == nullptr) {
        return nullptr;
    }
    const auto & tensors = llama_internal_get_tensor_map(model);
    if (index >= tensors.size()) {
        return nullptr;
    }
    return tensors[index].second;
}

}  // extern "C"
