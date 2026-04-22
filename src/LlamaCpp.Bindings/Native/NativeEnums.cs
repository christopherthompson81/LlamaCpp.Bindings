// Enum mirrors for llama.h / ggml.h. C enums are sized as int (4 bytes) on every
// platform llama.cpp targets. Names/values match the pinned header exactly —
// edits here must be paired with tools/struct-sizes.json after a header bump.

namespace LlamaCpp.Bindings.Native;

internal enum llama_vocab_type : int
{
    LLAMA_VOCAB_TYPE_NONE   = 0,
    LLAMA_VOCAB_TYPE_SPM    = 1,
    LLAMA_VOCAB_TYPE_BPE    = 2,
    LLAMA_VOCAB_TYPE_WPM    = 3,
    LLAMA_VOCAB_TYPE_UGM    = 4,
    LLAMA_VOCAB_TYPE_RWKV   = 5,
    LLAMA_VOCAB_TYPE_PLAMO2 = 6,
}

[Flags]
internal enum llama_token_attr : int
{
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
}

internal enum llama_rope_type : int
{
    LLAMA_ROPE_TYPE_NONE   = -1,
    LLAMA_ROPE_TYPE_NORM   = 0,
    // Values below come from ggml.h's GGML_ROPE_TYPE_* #defines. If ggml's
    // numbering ever changes, the xref pipeline will surface it via the
    // llama_rope_type diff.
    LLAMA_ROPE_TYPE_NEOX   = 2,
    LLAMA_ROPE_TYPE_MROPE  = 8,
    LLAMA_ROPE_TYPE_VISION = 24,
    LLAMA_ROPE_TYPE_IMROPE = 40,
}

internal enum llama_rope_scaling_type : int
{
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
}

internal enum llama_pooling_type : int
{
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE        = 0,
    LLAMA_POOLING_TYPE_MEAN        = 1,
    LLAMA_POOLING_TYPE_CLS         = 2,
    LLAMA_POOLING_TYPE_LAST        = 3,
    LLAMA_POOLING_TYPE_RANK        = 4,
}

internal enum llama_attention_type : int
{
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
}

internal enum llama_flash_attn_type : int
{
    LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
    LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
}

internal enum llama_split_mode : int
{
    LLAMA_SPLIT_MODE_NONE  = 0,
    LLAMA_SPLIT_MODE_LAYER = 1,
    LLAMA_SPLIT_MODE_ROW   = 2,
}

// From ggml.h — referenced by llama_context_params.type_k / type_v.
internal enum ggml_type : int
{
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
}

// From ggml.h — consumed by llama_log_set callback and llama_params_fit.
internal enum ggml_log_level : int
{
    GGML_LOG_LEVEL_NONE  = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO  = 2,
    GGML_LOG_LEVEL_WARN  = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT  = 5,
}
