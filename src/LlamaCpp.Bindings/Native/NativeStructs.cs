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

// ----- Multimodal (mtmd.h) -----

[StructLayout(LayoutKind.Sequential)]
internal struct mtmd_context_params
{
    // 0   | bool use_gpu
    [MarshalAs(UnmanagedType.I1)] public bool use_gpu;
    // 1   | bool print_timings
    [MarshalAs(UnmanagedType.I1)] public bool print_timings;
    // 2-3 | padding
    // 4   | int n_threads
    public int n_threads;
    // 8   | const char * image_marker — deprecated, prefer media_marker.
    public IntPtr image_marker;
    // 16  | const char * media_marker
    public IntPtr media_marker;
    // 24  | enum llama_flash_attn_type flash_attn_type
    public llama_flash_attn_type flash_attn_type;
    // 28  | bool warmup
    [MarshalAs(UnmanagedType.I1)] public bool warmup;
    // 29-31 | padding
    // 32  | int image_min_tokens
    public int image_min_tokens;
    // 36  | int image_max_tokens
    public int image_max_tokens;
    // 40  | ggml_backend_sched_eval_callback cb_eval — fn ptr, IntPtr.Zero ok.
    public IntPtr cb_eval;
    // 48  | void * cb_eval_user_data
    public IntPtr cb_eval_user_data;

    public const int ExpectedSize = 56;
}

[StructLayout(LayoutKind.Sequential)]
internal struct mtmd_input_text
{
    // 0  | const char * text — UTF-8, caller-owned. We pin a byte span and
    //      assign the address directly, so this stays as IntPtr rather than a
    //      marshaled string.
    public IntPtr text;
    // 8  | bool add_special
    [MarshalAs(UnmanagedType.I1)] public bool add_special;
    // 9  | bool parse_special
    [MarshalAs(UnmanagedType.I1)] public bool parse_special;
    // 10-15 | padding

    public const int ExpectedSize = 16;
}

/// <summary>
/// Position tuple for M-RoPE (multi-dimensional rotary position embedding)
/// vision models. Each image token gets a (t, x, y) coordinate rather than a
/// single sequential index; the text side's attention uses these to attend
/// to image patches with proper 2D spatial awareness.
/// </summary>
/// <remarks>
/// <c>z</c> is reserved by upstream for future use (volumetric media?).
/// Returned by-value from <c>mtmd_image_tokens_get_decoder_pos</c>.
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
internal struct mtmd_decoder_pos
{
    public uint t;   // 0
    public uint x;   // 4
    public uint y;   // 8
    public uint z;   // 12 — reserved

    public const int ExpectedSize = 16;
}

// llama_model_quantize_params — passed by pointer to llama_model_quantize.
// Field order is locked to llama.h. The 6 packed bools sit at offsets 16..21
// with a 2-byte tail pad before the pointer block at offset 24.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_model_quantize_params
{
    // 0  | int32_t nthread — <=0 means hardware_concurrency()
    public int nthread;
    // 4  | enum llama_ftype ftype
    public llama_ftype ftype;
    // 8  | enum ggml_type output_tensor_type
    public ggml_type output_tensor_type;
    // 12 | enum ggml_type token_embedding_type
    public ggml_type token_embedding_type;
    // 16 | bool allow_requantize
    [MarshalAs(UnmanagedType.I1)] public bool allow_requantize;
    // 17 | bool quantize_output_tensor
    [MarshalAs(UnmanagedType.I1)] public bool quantize_output_tensor;
    // 18 | bool only_copy
    [MarshalAs(UnmanagedType.I1)] public bool only_copy;
    // 19 | bool pure
    [MarshalAs(UnmanagedType.I1)] public bool pure;
    // 20 | bool keep_split
    [MarshalAs(UnmanagedType.I1)] public bool keep_split;
    // 21 | bool dry_run
    [MarshalAs(UnmanagedType.I1)] public bool dry_run;
    // 22..23 | (2 bytes padding to align next pointer to 8)
    // 24 | const struct llama_model_imatrix_data * imatrix
    public IntPtr imatrix;
    // 32 | const struct llama_model_kv_override * kv_overrides
    public IntPtr kv_overrides;
    // 40 | const struct llama_model_tensor_override * tt_overrides
    public IntPtr tt_overrides;
    // 48 | const int32_t * prune_layers — null-terminator-free count is
    //      caller's responsibility (the native side reads until... see
    //      llama-quantize CLI for the exact convention).
    public IntPtr prune_layers;

    public const int ExpectedSize = 56;
}

// llama_model_imatrix_data — single imatrix entry (one tensor's column-sum).
// llama_model_quantize takes a pointer to one of these; in practice the
// CLI builds a flat array of them, but the struct itself is just three
// pointer-sized words.
[StructLayout(LayoutKind.Sequential)]
internal struct llama_model_imatrix_data
{
    // 0  | const char * name — tensor name the data is for
    public IntPtr name;
    // 8  | const float * data — column-importance values
    public IntPtr data;
    // 16 | size_t size — number of floats at @data
    public nuint size;

    public const int ExpectedSize = 24;
}

// llama_model_tensor_override — pattern → ggml_type. Used by quantize to
// pin specific tensors to a non-default type (e.g. keep output.weight at F16
// regardless of the chosen ftype).
[StructLayout(LayoutKind.Sequential)]
internal struct llama_model_tensor_override
{
    // 0  | const char * pattern — fnmatch-style pattern over tensor names
    public IntPtr pattern;
    // 8  | enum ggml_type type — target type for matching tensors
    public ggml_type type;
    // 12..15 | (4 bytes tail pad — struct is 8-aligned by the leading pointer)

    public const int ExpectedSize = 16;
}

// llama_model_kv_override — overrides one GGUF metadata key when loading or
// quantizing. The C struct is { enum tag; char key[128]; union { ... }; }
// where the union's 128-byte char[] member dominates and is 8-byte aligned,
// giving 4 + 128 + 4-pad + 128 = 264 bytes total.
//
// We don't currently need to construct these from C# — quantize callers can
// pass IntPtr.Zero for kv_overrides — but the struct must mirror correctly
// so future callers can build arrays without re-deriving the layout. The
// inline char buffers are exposed as fixed-size byte arrays; encoders should
// write UTF-8 with a trailing NUL byte and refuse keys/strings >= 128 bytes
// of UTF-8 (matches the native limit).
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct llama_model_kv_override
{
    // 0   | enum llama_model_kv_override_type tag
    public llama_model_kv_override_type tag;
    // 4   | char key[128] — NUL-terminated UTF-8 key
    public fixed byte key[128];
    // 132..135 | explicit 4-byte pad: the union's int64/double members force
    //            8-byte alignment of the union body in C, but `fixed byte[]`
    //            is 1-byte-aligned in C#, so we need to materialise the pad.
    private fixed byte _pad[4];
    // 136 | union { int64; double; bool; char[128] }; the char[] dominates
    //       the size, so we mirror it and treat the other reads as
    //       overlapping prefixes (val_str[0..7] aliases val_i64 etc.).
    public fixed byte val_str[128];

    public const int ExpectedSize = 264;
}

/// <summary>
/// Mirror of <c>ggml_tensor</c> from <c>ggml.h</c>. We only read this from
/// inside the imatrix eval-callback — the runtime hands us a raw
/// <c>ggml_tensor *</c> and we need byte-correct field offsets to pull
/// type/op/src/data/name/dims out without a P/Invoke per access.
/// </summary>
/// <remarks>
/// <para>
/// The struct is 336 bytes on 64-bit Linux/macOS/Windows. Layout:
/// <c>type(0,4)</c>, pad(4..7), <c>buffer(8,8)</c>, <c>ne[4](16,32)</c>,
/// <c>nb[4](48,32)</c>, <c>op(80,4)</c>, <c>op_params[16](84,64)</c>,
/// <c>flags(148,4)</c>, <c>src[10](152,80)</c>, <c>view_src(232,8)</c>,
/// <c>view_offs(240,8)</c>, <c>data(248,8)</c>, <c>name[64](256,64)</c>,
/// <c>extra(320,8)</c>, <c>padding[8](328,8)</c>.
/// </para>
/// <para>
/// Field-offset assertions in <see cref="StructLayoutTests"/> guard against
/// silent header drift; if any of them fire after a llama.cpp bump, re-run
/// <c>tools/dump-struct-sizes.sh</c> and reconcile here before believing
/// any imatrix output.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct ggml_tensor
{
    // 0   | enum ggml_type type
    public ggml_type type;
    // 4   | (4 bytes pad to align next pointer to 8)
    private int _pad_after_type;
    // 8   | struct ggml_backend_buffer * buffer
    public IntPtr buffer;
    // 16  | int64_t ne[4]
    public fixed long ne[4];
    // 48  | size_t nb[4]
    public fixed ulong nb[4];
    // 80  | enum ggml_op op
    public ggml_op op;
    // 84  | int32_t op_params[16] — reserved 64 bytes; contents are
    //       op-specific and we don't read them.
    public fixed int op_params[16];
    // 148 | int32_t flags
    public int flags;
    // 152 | struct ggml_tensor * src[10]
    public fixed long src[10]; // pointer-sized; we read these as IntPtrs.
    // 232 | struct ggml_tensor * view_src
    public IntPtr view_src;
    // 240 | size_t view_offs
    public nuint view_offs;
    // 248 | void * data
    public IntPtr data;
    // 256 | char name[64]
    public fixed byte name[64];
    // 320 | void * extra
    public IntPtr extra;
    // 328 | char padding[8]
    public fixed byte padding[8];

    public const int ExpectedSize = 336;
}

/// <summary>
/// Mirror of <c>gguf_init_params</c>. The 1-byte bool is followed by 7 bytes
/// of padding before the pointer; <see cref="LayoutKind.Sequential"/> with
/// the default Pack=8 reproduces that layout under .NET on 64-bit
/// platforms.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct gguf_init_params
{
    [MarshalAs(UnmanagedType.I1)] public bool no_alloc;
    public IntPtr ctx; // ggml_context** — pass IntPtr.Zero for "don't allocate"

    public const int ExpectedSize = 16;
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
        Check<mtmd_context_params>(mtmd_context_params.ExpectedSize);
        Check<mtmd_input_text>(mtmd_input_text.ExpectedSize);
        Check<mtmd_decoder_pos>(mtmd_decoder_pos.ExpectedSize);
        Check<gguf_init_params>(gguf_init_params.ExpectedSize);
        Check<llama_model_quantize_params>(llama_model_quantize_params.ExpectedSize);
        Check<llama_model_imatrix_data>(llama_model_imatrix_data.ExpectedSize);
        Check<llama_model_tensor_override>(llama_model_tensor_override.ExpectedSize);
        Check<llama_model_kv_override>(llama_model_kv_override.ExpectedSize);
        Check<ggml_tensor>(ggml_tensor.ExpectedSize);
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
