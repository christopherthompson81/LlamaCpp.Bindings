// Emits sizeof() and selected field offsets for every struct we mirror in C#.
// Build against the pinned llama.h (or any matching llama.cpp include dir).
// Output: JSON on stdout consumed by tools/dump-struct-sizes.sh and baked
// into the StructLayoutTests as ground truth.
//
// We don't hard-code paths — caller passes -I<llama.cpp include dir>.

#include <stddef.h>
#include <stdio.h>

#include "llama.h"

#define SZ(T)      printf("    \"sizeof_" #T "\": %zu,\n",      sizeof(T))
#define OFF(T, F)  printf("    \"offsetof_" #T "__" #F "\": %zu,\n", offsetof(T, F))

int main(void) {
    printf("{\n");

    // ----- llama_model_params -----
    SZ(struct llama_model_params);
    OFF(struct llama_model_params, devices);
    OFF(struct llama_model_params, tensor_buft_overrides);
    OFF(struct llama_model_params, n_gpu_layers);
    OFF(struct llama_model_params, split_mode);
    OFF(struct llama_model_params, main_gpu);
    OFF(struct llama_model_params, tensor_split);
    OFF(struct llama_model_params, progress_callback);
    OFF(struct llama_model_params, progress_callback_user_data);
    OFF(struct llama_model_params, kv_overrides);
    OFF(struct llama_model_params, vocab_only);
    OFF(struct llama_model_params, use_mmap);
    OFF(struct llama_model_params, use_direct_io);
    OFF(struct llama_model_params, use_mlock);
    OFF(struct llama_model_params, check_tensors);
    OFF(struct llama_model_params, use_extra_bufts);
    OFF(struct llama_model_params, no_host);
    OFF(struct llama_model_params, no_alloc);

    // ----- llama_context_params -----
    SZ(struct llama_context_params);
    OFF(struct llama_context_params, n_ctx);
    OFF(struct llama_context_params, n_batch);
    OFF(struct llama_context_params, n_ubatch);
    OFF(struct llama_context_params, n_seq_max);
    OFF(struct llama_context_params, n_threads);
    OFF(struct llama_context_params, n_threads_batch);
    OFF(struct llama_context_params, rope_scaling_type);
    OFF(struct llama_context_params, pooling_type);
    OFF(struct llama_context_params, attention_type);
    OFF(struct llama_context_params, flash_attn_type);
    OFF(struct llama_context_params, rope_freq_base);
    OFF(struct llama_context_params, rope_freq_scale);
    OFF(struct llama_context_params, yarn_ext_factor);
    OFF(struct llama_context_params, yarn_attn_factor);
    OFF(struct llama_context_params, yarn_beta_fast);
    OFF(struct llama_context_params, yarn_beta_slow);
    OFF(struct llama_context_params, yarn_orig_ctx);
    OFF(struct llama_context_params, defrag_thold);
    OFF(struct llama_context_params, cb_eval);
    OFF(struct llama_context_params, cb_eval_user_data);
    OFF(struct llama_context_params, type_k);
    OFF(struct llama_context_params, type_v);
    OFF(struct llama_context_params, abort_callback);
    OFF(struct llama_context_params, abort_callback_data);
    OFF(struct llama_context_params, embeddings);
    OFF(struct llama_context_params, offload_kqv);
    OFF(struct llama_context_params, no_perf);
    OFF(struct llama_context_params, op_offload);
    OFF(struct llama_context_params, swa_full);
    OFF(struct llama_context_params, kv_unified);
    OFF(struct llama_context_params, samplers);
    OFF(struct llama_context_params, n_samplers);

    // ----- llama_chat_message -----
    SZ(struct llama_chat_message);
    OFF(struct llama_chat_message, role);
    OFF(struct llama_chat_message, content);

    // ----- llama_batch -----
    SZ(struct llama_batch);
    OFF(struct llama_batch, n_tokens);
    OFF(struct llama_batch, token);
    OFF(struct llama_batch, embd);
    OFF(struct llama_batch, pos);
    OFF(struct llama_batch, n_seq_id);
    OFF(struct llama_batch, seq_id);
    OFF(struct llama_batch, logits);

    // ----- llama_sampler_chain_params -----
    SZ(struct llama_sampler_chain_params);
    OFF(struct llama_sampler_chain_params, no_perf);

    // ----- llama_perf_context_data -----
    SZ(struct llama_perf_context_data);
    OFF(struct llama_perf_context_data, t_start_ms);
    OFF(struct llama_perf_context_data, t_load_ms);
    OFF(struct llama_perf_context_data, t_p_eval_ms);
    OFF(struct llama_perf_context_data, t_eval_ms);
    OFF(struct llama_perf_context_data, n_p_eval);
    OFF(struct llama_perf_context_data, n_eval);
    OFF(struct llama_perf_context_data, n_reused);

    // ----- llama_perf_sampler_data -----
    SZ(struct llama_perf_sampler_data);
    OFF(struct llama_perf_sampler_data, t_sample_ms);
    OFF(struct llama_perf_sampler_data, n_sample);

    // ----- llama_logit_bias -----
    SZ(struct llama_logit_bias);
    OFF(struct llama_logit_bias, token);
    OFF(struct llama_logit_bias, bias);

    // Terminator to keep JSON valid without worrying about trailing comma.
    printf("    \"_end\": 0\n");
    printf("}\n");
    return 0;
}
