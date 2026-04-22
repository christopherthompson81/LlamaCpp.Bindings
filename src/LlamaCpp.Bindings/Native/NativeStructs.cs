// Struct mirrors for llama.h. These must match the C layout byte-for-byte.
// Ground truth for sizes/offsets lives in tools/struct-sizes.json, produced
// from tools/dump-struct-sizes.c against the pinned header. The size assertions
// below run at module init — if they fire, we refuse to load.
//
// Rules (see CLAUDE.md):
//   - [StructLayout(LayoutKind.Sequential)] always; never Auto.
//   - Field order matches llama.h exactly; do not reorder for readability.
//   - nint/nuint for pointer-sized integers; IntPtr for pointers passed through.
//   - [MarshalAs(UnmanagedType.I1)] bool for C _Bool (1 byte).
//   - C enums are 4-byte ints; we mirror them as `int`-backed C# enums above.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native;

[StructLayout(LayoutKind.Sequential)]
internal struct llama_model_params
{
    // 0  | ggml_backend_dev_t * devices — NULL-terminated device list (opaque pointer).
    public IntPtr devices;

    // 8  | const struct llama_model_tensor_buft_override * tensor_buft_overrides
    public IntPtr tensor_buft_overrides;

    // 16 | int32_t n_gpu_layers
    public int n_gpu_layers;

    // 20 | enum llama_split_mode split_mode
    public llama_split_mode split_mode;

    // 24 | int32_t main_gpu
    public int main_gpu;

    // 28 | (4 bytes padding before next 8-byte-aligned pointer)

    // 32 | const float * tensor_split
    public IntPtr tensor_split;

    // 40 | llama_progress_callback progress_callback — function pointer.
    public IntPtr progress_callback;

    // 48 | void * progress_callback_user_data
    public IntPtr progress_callback_user_data;

    // 56 | const struct llama_model_kv_override * kv_overrides
    public IntPtr kv_overrides;

    // 64..71 | 8 bools packed
    [MarshalAs(UnmanagedType.I1)] public bool vocab_only;
    [MarshalAs(UnmanagedType.I1)] public bool use_mmap;
    [MarshalAs(UnmanagedType.I1)] public bool use_direct_io;
    [MarshalAs(UnmanagedType.I1)] public bool use_mlock;
    [MarshalAs(UnmanagedType.I1)] public bool check_tensors;
    [MarshalAs(UnmanagedType.I1)] public bool use_extra_bufts;
    [MarshalAs(UnmanagedType.I1)] public bool no_host;
    [MarshalAs(UnmanagedType.I1)] public bool no_alloc;

    public const int ExpectedSize = 72;
}

[StructLayout(LayoutKind.Sequential)]
internal struct llama_context_params
{
    // 0   | uint32_t n_ctx            — text context, 0 = from model
    public uint n_ctx;
    // 4   | uint32_t n_batch          — logical maximum batch submitted to llama_decode
    public uint n_batch;
    // 8   | uint32_t n_ubatch         — physical maximum batch size
    public uint n_ubatch;
    // 12  | uint32_t n_seq_max        — max distinct sequences (recurrent models)
    public uint n_seq_max;
    // 16  | int32_t  n_threads
    public int n_threads;
    // 20  | int32_t  n_threads_batch
    public int n_threads_batch;

    // 24  | enum llama_rope_scaling_type rope_scaling_type
    public llama_rope_scaling_type rope_scaling_type;
    // 28  | enum llama_pooling_type pooling_type
    public llama_pooling_type pooling_type;
    // 32  | enum llama_attention_type attention_type
    public llama_attention_type attention_type;
    // 36  | enum llama_flash_attn_type flash_attn_type
    public llama_flash_attn_type flash_attn_type;

    // 40  | float rope_freq_base      — 0 = from model
    public float rope_freq_base;
    // 44  | float rope_freq_scale     — 0 = from model
    public float rope_freq_scale;
    // 48  | float yarn_ext_factor     — negative = from model
    public float yarn_ext_factor;
    // 52  | float yarn_attn_factor
    public float yarn_attn_factor;
    // 56  | float yarn_beta_fast
    public float yarn_beta_fast;
    // 60  | float yarn_beta_slow
    public float yarn_beta_slow;
    // 64  | uint32_t yarn_orig_ctx
    public uint yarn_orig_ctx;
    // 68  | float defrag_thold        — [DEPRECATED]
    public float defrag_thold;

    // 72  | ggml_backend_sched_eval_callback cb_eval — function pointer
    public IntPtr cb_eval;
    // 80  | void * cb_eval_user_data
    public IntPtr cb_eval_user_data;

    // 88  | enum ggml_type type_k — KV cache K data type
    public ggml_type type_k;
    // 92  | enum ggml_type type_v — KV cache V data type
    public ggml_type type_v;

    // 96  | ggml_abort_callback abort_callback — function pointer
    public IntPtr abort_callback;
    // 104 | void * abort_callback_data
    public IntPtr abort_callback_data;

    // 112..117 | 6 bools packed
    [MarshalAs(UnmanagedType.I1)] public bool embeddings;
    [MarshalAs(UnmanagedType.I1)] public bool offload_kqv;
    [MarshalAs(UnmanagedType.I1)] public bool no_perf;
    [MarshalAs(UnmanagedType.I1)] public bool op_offload;
    [MarshalAs(UnmanagedType.I1)] public bool swa_full;
    [MarshalAs(UnmanagedType.I1)] public bool kv_unified;

    // 118..119 | (2 bytes padding before next 8-byte-aligned pointer)

    // 120 | struct llama_sampler_seq_config * samplers
    public IntPtr samplers;
    // 128 | size_t n_samplers
    public nuint n_samplers;

    public const int ExpectedSize = 136;
}

// llama_chat_message: two const char* pointers (role, content). Both own by
// the caller — llama.cpp only reads them during llama_chat_apply_template.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_chat_message
{
    public IntPtr role;
    public IntPtr content;

    public const int ExpectedSize = 16; // two 8-byte pointers on 64-bit
}

// llama_batch: input buffer for llama_decode / llama_encode.
// Most callers use llama_batch_get_one which stitches the first few pointer
// fields at the tokens array and leaves the rest NULL — llama_decode fills in
// positions/seq_ids/logits defaults automatically in that mode.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_batch
{
    // 0  | int32_t n_tokens
    public int n_tokens;
    // 4  | (4 bytes padding before next 8-byte-aligned pointer)
    // 8  | llama_token * token  (int32_t*)
    public IntPtr token;
    // 16 | float * embd
    public IntPtr embd;
    // 24 | llama_pos * pos  (int32_t*)
    public IntPtr pos;
    // 32 | int32_t * n_seq_id
    public IntPtr n_seq_id;
    // 40 | llama_seq_id ** seq_id
    public IntPtr seq_id;
    // 48 | int8_t * logits (aka "output")
    public IntPtr logits;

    public const int ExpectedSize = 56;
}

// llama_sampler_chain_params: single-field struct, one 1-byte bool.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_sampler_chain_params
{
    [MarshalAs(UnmanagedType.I1)] public bool no_perf;

    public const int ExpectedSize = 1;
}

// llama_perf_context_data: timing + count info from a context.
// 4 doubles + 3 int32 = 44 bytes data + 4 bytes struct-alignment padding = 48.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_perf_context_data
{
    public double t_start_ms;    // absolute start time
    public double t_load_ms;     // time needed for loading the model
    public double t_p_eval_ms;   // time spent processing the prompt
    public double t_eval_ms;     // time spent generating tokens
    public int n_p_eval;         // number of prompt tokens evaluated
    public int n_eval;           // number of generated tokens
    public int n_reused;         // number of times a compute graph was reused

    public const int ExpectedSize = 48;
}

// llama_perf_sampler_data: timing + count info from a sampler chain.
// 1 double + 1 int32 = 12 bytes data + 4 bytes struct-alignment padding = 16.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_perf_sampler_data
{
    public double t_sample_ms;   // total time spent in sampling
    public int n_sample;         // number of tokens sampled

    public const int ExpectedSize = 16;
}

// llama_logit_bias: (token, bias) pair. 4 + 4 = 8 bytes, no padding.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_logit_bias
{
    public int token;
    public float bias;

    public const int ExpectedSize = 8;
}

// llama_token_data: one candidate in the logit/prob distribution. 12 bytes.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_token_data
{
    public int id;      // 0  token id
    public float logit; // 4  log-odds
    public float p;     // 8  probability
    // = 12 bytes, no padding

    public const int ExpectedSize = 12;
}

// llama_token_data_array: the candidate set passed to llama_sampler_apply.
// Samplers mutate this in place — .size can shrink, .data pointer may be
// reassigned (though most samplers edit in-place without reallocating).
[StructLayout(LayoutKind.Sequential)]
internal struct llama_token_data_array
{
    public IntPtr data;     // 0   llama_token_data*
    public nuint size;      // 8   size_t
    public long selected;   // 16  int64_t (index into data, not token id)
    [MarshalAs(UnmanagedType.I1)] public bool sorted; // 24
    // 7 bytes tail padding for struct alignment

    public const int ExpectedSize = 32;
}

/// <summary>
/// Struct layout assertions. Called once from <see cref="LlamaBackend"/>'s
/// static constructor before any native call is made. If the native struct
/// size has drifted from our C# mirror, we throw here rather than let the
/// next P/Invoke corrupt memory.
/// </summary>
internal static class NativeLayout
{
    public static void Verify()
    {
        Check<llama_model_params>(llama_model_params.ExpectedSize);
        Check<llama_context_params>(llama_context_params.ExpectedSize);
        Check<llama_chat_message>(llama_chat_message.ExpectedSize);
        Check<llama_batch>(llama_batch.ExpectedSize);
        Check<llama_sampler_chain_params>(llama_sampler_chain_params.ExpectedSize);
        Check<llama_perf_context_data>(llama_perf_context_data.ExpectedSize);
        Check<llama_perf_sampler_data>(llama_perf_sampler_data.ExpectedSize);
        Check<llama_logit_bias>(llama_logit_bias.ExpectedSize);
        Check<llama_token_data>(llama_token_data.ExpectedSize);
        Check<llama_token_data_array>(llama_token_data_array.ExpectedSize);
    }

    private static void Check<T>(int expected, [CallerMemberName] string? caller = null) where T : struct
    {
        _ = caller;
        var actual = Unsafe.SizeOf<T>();
        if (actual != expected)
        {
            throw new InvalidOperationException(
                $"Struct layout mismatch: {typeof(T).Name} is {actual} bytes in managed code " +
                $"but the pinned llama.h expects {expected}. The native binary and the pinned " +
                $"header have drifted — re-run tools/dump-struct-sizes.sh and update the mirror " +
                $"before loading the native library.");
        }
    }
}
