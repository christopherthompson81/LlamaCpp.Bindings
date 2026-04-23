using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Verifies that our managed struct mirrors match the pinned C layout.
/// Ground truth values come from tools/struct-sizes.json (which is produced
/// by compiling tools/dump-struct-sizes.c against the pinned llama.h). If these
/// ever fail after a header bump, regenerate the JSON first, then reconcile.
/// </summary>
public class StructLayoutTests
{
    [Fact]
    public void LlamaModelParams_Size_Matches_Pinned()
    {
        Assert.Equal(72, Unsafe.SizeOf<llama_model_params>());
    }

    [Fact]
    public void LlamaContextParams_Size_Matches_Pinned()
    {
        Assert.Equal(136, Unsafe.SizeOf<llama_context_params>());
    }

    [Theory]
    [InlineData(nameof(llama_model_params.devices),                     0)]
    [InlineData(nameof(llama_model_params.tensor_buft_overrides),       8)]
    [InlineData(nameof(llama_model_params.n_gpu_layers),               16)]
    [InlineData(nameof(llama_model_params.split_mode),                 20)]
    [InlineData(nameof(llama_model_params.main_gpu),                   24)]
    [InlineData(nameof(llama_model_params.tensor_split),               32)]
    [InlineData(nameof(llama_model_params.progress_callback),          40)]
    [InlineData(nameof(llama_model_params.progress_callback_user_data), 48)]
    [InlineData(nameof(llama_model_params.kv_overrides),               56)]
    [InlineData(nameof(llama_model_params.vocab_only),                 64)]
    [InlineData(nameof(llama_model_params.use_mmap),                   65)]
    [InlineData(nameof(llama_model_params.use_direct_io),              66)]
    [InlineData(nameof(llama_model_params.use_mlock),                  67)]
    [InlineData(nameof(llama_model_params.check_tensors),              68)]
    [InlineData(nameof(llama_model_params.use_extra_bufts),            69)]
    [InlineData(nameof(llama_model_params.no_host),                    70)]
    [InlineData(nameof(llama_model_params.no_alloc),                   71)]
    public void LlamaModelParams_Field_Offsets_Match_Pinned(string field, int expected)
    {
        Assert.Equal(expected, Marshal.OffsetOf<llama_model_params>(field).ToInt32());
    }

    [Theory]
    [InlineData(nameof(llama_context_params.n_ctx),               0)]
    [InlineData(nameof(llama_context_params.n_batch),             4)]
    [InlineData(nameof(llama_context_params.n_ubatch),            8)]
    [InlineData(nameof(llama_context_params.n_seq_max),          12)]
    [InlineData(nameof(llama_context_params.n_threads),          16)]
    [InlineData(nameof(llama_context_params.n_threads_batch),    20)]
    [InlineData(nameof(llama_context_params.rope_scaling_type),  24)]
    [InlineData(nameof(llama_context_params.pooling_type),       28)]
    [InlineData(nameof(llama_context_params.attention_type),     32)]
    [InlineData(nameof(llama_context_params.flash_attn_type),    36)]
    [InlineData(nameof(llama_context_params.rope_freq_base),     40)]
    [InlineData(nameof(llama_context_params.rope_freq_scale),    44)]
    [InlineData(nameof(llama_context_params.yarn_ext_factor),    48)]
    [InlineData(nameof(llama_context_params.yarn_attn_factor),   52)]
    [InlineData(nameof(llama_context_params.yarn_beta_fast),     56)]
    [InlineData(nameof(llama_context_params.yarn_beta_slow),     60)]
    [InlineData(nameof(llama_context_params.yarn_orig_ctx),      64)]
    [InlineData(nameof(llama_context_params.defrag_thold),       68)]
    [InlineData(nameof(llama_context_params.cb_eval),            72)]
    [InlineData(nameof(llama_context_params.cb_eval_user_data),  80)]
    [InlineData(nameof(llama_context_params.type_k),             88)]
    [InlineData(nameof(llama_context_params.type_v),             92)]
    [InlineData(nameof(llama_context_params.abort_callback),     96)]
    [InlineData(nameof(llama_context_params.abort_callback_data), 104)]
    [InlineData(nameof(llama_context_params.embeddings),         112)]
    [InlineData(nameof(llama_context_params.offload_kqv),        113)]
    [InlineData(nameof(llama_context_params.no_perf),            114)]
    [InlineData(nameof(llama_context_params.op_offload),         115)]
    [InlineData(nameof(llama_context_params.swa_full),           116)]
    [InlineData(nameof(llama_context_params.kv_unified),         117)]
    [InlineData(nameof(llama_context_params.samplers),           120)]
    [InlineData(nameof(llama_context_params.n_samplers),         128)]
    public void LlamaContextParams_Field_Offsets_Match_Pinned(string field, int expected)
    {
        Assert.Equal(expected, Marshal.OffsetOf<llama_context_params>(field).ToInt32());
    }

    [Fact]
    public void MtmdContextParams_Size_Matches_Pinned()
    {
        Assert.Equal(56, Unsafe.SizeOf<mtmd_context_params>());
    }

    [Theory]
    [InlineData(nameof(mtmd_context_params.use_gpu),            0)]
    [InlineData(nameof(mtmd_context_params.print_timings),      1)]
    [InlineData(nameof(mtmd_context_params.n_threads),          4)]
    [InlineData(nameof(mtmd_context_params.image_marker),       8)]
    [InlineData(nameof(mtmd_context_params.media_marker),      16)]
    [InlineData(nameof(mtmd_context_params.flash_attn_type),   24)]
    [InlineData(nameof(mtmd_context_params.warmup),            28)]
    [InlineData(nameof(mtmd_context_params.image_min_tokens),  32)]
    [InlineData(nameof(mtmd_context_params.image_max_tokens),  36)]
    [InlineData(nameof(mtmd_context_params.cb_eval),           40)]
    [InlineData(nameof(mtmd_context_params.cb_eval_user_data), 48)]
    public void MtmdContextParams_Field_Offsets_Match_Pinned(string field, int expected)
    {
        Assert.Equal(expected, Marshal.OffsetOf<mtmd_context_params>(field).ToInt32());
    }

    [Fact]
    public void MtmdInputText_Size_Matches_Pinned()
    {
        Assert.Equal(16, Unsafe.SizeOf<mtmd_input_text>());
    }

    [Theory]
    [InlineData(nameof(mtmd_input_text.text),          0)]
    [InlineData(nameof(mtmd_input_text.add_special),   8)]
    [InlineData(nameof(mtmd_input_text.parse_special), 9)]
    public void MtmdInputText_Field_Offsets_Match_Pinned(string field, int expected)
    {
        Assert.Equal(expected, Marshal.OffsetOf<mtmd_input_text>(field).ToInt32());
    }

    [Fact]
    public void NativeLayout_Verify_Does_Not_Throw()
    {
        // If this throws the binding refuses to load, so it must pass whenever
        // the test project builds against the pinned native library.
        NativeLayout.Verify();
    }
}
