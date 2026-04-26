// Emits sizeof() and selected field offsets for every struct we mirror in C#.
// Build against the pinned llama.h (or any matching llama.cpp include dir).
// Output: JSON on stdout consumed by tools/dump-struct-sizes.sh and baked
// into the StructLayoutTests as ground truth.
//
// We don't hard-code paths — caller passes -I<llama.cpp include dir>.

#include <stddef.h>
#include <stdio.h>

#include "llama.h"
#include "mtmd.h"
#include "ggml.h"

#define SZ(T)      printf("    \"sizeof_" #T "\": %zu,\n",      sizeof(T))
#define OFF(T, F)  printf("    \"offsetof_" #T "__" #F "\": %zu,\n", offsetof(T, F))
#define ENUMV(E)   printf("    \"enumvalue_" #E "\": %d,\n",      (int)(E))

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

    // ----- llama_token_data -----
    SZ(llama_token_data);
    OFF(llama_token_data, id);
    OFF(llama_token_data, logit);
    OFF(llama_token_data, p);

    // ----- llama_token_data_array -----
    SZ(llama_token_data_array);
    OFF(llama_token_data_array, data);
    OFF(llama_token_data_array, size);
    OFF(llama_token_data_array, selected);
    OFF(llama_token_data_array, sorted);

    // ----- mtmd_context_params -----
    SZ(struct mtmd_context_params);
    OFF(struct mtmd_context_params, use_gpu);
    OFF(struct mtmd_context_params, print_timings);
    OFF(struct mtmd_context_params, n_threads);
    OFF(struct mtmd_context_params, image_marker);
    OFF(struct mtmd_context_params, media_marker);
    OFF(struct mtmd_context_params, flash_attn_type);
    OFF(struct mtmd_context_params, warmup);
    OFF(struct mtmd_context_params, image_min_tokens);
    OFF(struct mtmd_context_params, image_max_tokens);
    OFF(struct mtmd_context_params, cb_eval);
    OFF(struct mtmd_context_params, cb_eval_user_data);

    // ----- mtmd_input_text -----
    SZ(struct mtmd_input_text);
    OFF(struct mtmd_input_text, text);
    OFF(struct mtmd_input_text, add_special);
    OFF(struct mtmd_input_text, parse_special);

    // ----- mtmd_decoder_pos ----- (added upstream b8893)
    SZ(struct mtmd_decoder_pos);
    OFF(struct mtmd_decoder_pos, t);
    OFF(struct mtmd_decoder_pos, x);
    OFF(struct mtmd_decoder_pos, y);
    OFF(struct mtmd_decoder_pos, z);

    // ----- llama_model_quantize_params -----
    SZ(struct llama_model_quantize_params);
    OFF(struct llama_model_quantize_params, nthread);
    OFF(struct llama_model_quantize_params, ftype);
    OFF(struct llama_model_quantize_params, output_tensor_type);
    OFF(struct llama_model_quantize_params, token_embedding_type);
    OFF(struct llama_model_quantize_params, allow_requantize);
    OFF(struct llama_model_quantize_params, quantize_output_tensor);
    OFF(struct llama_model_quantize_params, only_copy);
    OFF(struct llama_model_quantize_params, pure);
    OFF(struct llama_model_quantize_params, keep_split);
    OFF(struct llama_model_quantize_params, dry_run);
    OFF(struct llama_model_quantize_params, imatrix);
    OFF(struct llama_model_quantize_params, kv_overrides);
    OFF(struct llama_model_quantize_params, tt_overrides);
    OFF(struct llama_model_quantize_params, prune_layers);

    // ----- llama_model_imatrix_data -----
    SZ(struct llama_model_imatrix_data);
    OFF(struct llama_model_imatrix_data, name);
    OFF(struct llama_model_imatrix_data, data);
    OFF(struct llama_model_imatrix_data, size);

    // ----- llama_model_kv_override -----
    SZ(struct llama_model_kv_override);
    OFF(struct llama_model_kv_override, tag);
    OFF(struct llama_model_kv_override, key);

    // ----- llama_model_tensor_override -----
    SZ(struct llama_model_tensor_override);
    OFF(struct llama_model_tensor_override, pattern);
    OFF(struct llama_model_tensor_override, type);

    // ----- ggml_tensor (used by the imatrix eval-callback path) -----
    // The fields read from C# during the imatrix collector callback are
    // type / buffer / ne / nb / op / src / data / name. We assert sizes
    // and offsets for every field we read so a header bump that moves
    // anything trips StructLayoutTests rather than silently corrupting
    // memory inside the callback.
    SZ(struct ggml_tensor);
    OFF(struct ggml_tensor, type);
    OFF(struct ggml_tensor, buffer);
    OFF(struct ggml_tensor, ne);
    OFF(struct ggml_tensor, nb);
    OFF(struct ggml_tensor, op);
    OFF(struct ggml_tensor, op_params);
    OFF(struct ggml_tensor, flags);
    OFF(struct ggml_tensor, src);
    OFF(struct ggml_tensor, view_src);
    OFF(struct ggml_tensor, view_offs);
    OFF(struct ggml_tensor, data);
    OFF(struct ggml_tensor, name);
    OFF(struct ggml_tensor, extra);

    // ----- ggml_op values consumed by the imatrix callback -----
    ENUMV(GGML_OP_MUL_MAT);
    ENUMV(GGML_OP_MUL_MAT_ID);

    // ----- ggml_type_traits (used by the quantization-sensitivity sweep) -----
    SZ(struct ggml_type_traits);
    OFF(struct ggml_type_traits, type_name);
    OFF(struct ggml_type_traits, blck_size);
    OFF(struct ggml_type_traits, blck_size_interleave);
    OFF(struct ggml_type_traits, type_size);
    OFF(struct ggml_type_traits, is_quantized);
    OFF(struct ggml_type_traits, to_float);
    OFF(struct ggml_type_traits, from_float_ref);

    // Terminator to keep JSON valid without worrying about trailing comma.
    printf("    \"_end\": 0\n");
    printf("}\n");
    return 0;
}
