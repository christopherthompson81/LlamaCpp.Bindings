// P/Invoke declarations for llama.cpp. Rules (see CLAUDE.md):
//   - [LibraryImport("llama", ...)], never [DllImport].
//   - C# method names mirror the C name exactly (llama_decode, not LlamaDecode)
//     so header diffs apply mechanically.
//   - internal static partial; this class is never exposed to consumers.
//
// Phase 1 surface only: backend lifecycle, log routing, model load/free,
// context create/free, metadata accessors used by LlamaModel/LlamaContext.
// Further surface (tokenization, batch, decode, sampler chain) arrives in
// later phases.

using System.Runtime.InteropServices;

namespace LlamaCpp.Bindings.Native;

internal static partial class NativeMethods
{
    private const string LibName = "llama";
    private const string LibMtmd = "mtmd";

    // ----- Backend lifecycle -----

    [LibraryImport(LibName)]
    internal static partial void llama_backend_init();

    [LibraryImport(LibName)]
    internal static partial void llama_backend_free();

    // Callback: void (*)(ggml_log_level level, const char * text, void * user_data)
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GgmlLogCallback(ggml_log_level level, IntPtr text, IntPtr user_data);

    [LibraryImport(LibName)]
    internal static partial void llama_log_set(IntPtr log_callback, IntPtr user_data);

    // ----- Default param factories (return-by-value structs) -----

    [LibraryImport(LibName)]
    internal static partial llama_model_params llama_model_default_params();

    [LibraryImport(LibName)]
    internal static partial llama_context_params llama_context_default_params();

    // ----- Model load / metadata / free -----

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr llama_model_load_from_file(string path_model, llama_model_params params_);

    [LibraryImport(LibName)]
    internal static partial void llama_model_free(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_model_get_vocab(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_ctx_train(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_embd(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_layer(IntPtr model);

    // ----- Model metadata (Tier 1 expansion) -----

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_embd_inp(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_embd_out(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_head(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_head_kv(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_n_swa(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial float llama_model_rope_freq_scale_train(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial llama_rope_type llama_model_rope_type(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial ulong llama_model_size(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial ulong llama_model_n_params(IntPtr model);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_model_has_encoder(IntPtr model);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_model_has_decoder(IntPtr model);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_model_is_recurrent(IntPtr model);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_model_is_hybrid(IntPtr model);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_model_is_diffusion(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial int llama_model_decoder_start_token(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial uint llama_model_n_cls_out(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_model_cls_label(IntPtr model, uint i);

    // Probe-and-fill pattern: returns length on success, -1 on failure.
    // Caller allocates `buf` and passes `buf_size`. Null-terminated output.
    [LibraryImport(LibName)]
    internal static unsafe partial int llama_model_desc(IntPtr model, byte* buf, nuint buf_size);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int llama_model_meta_val_str(
        IntPtr model, string key, byte* buf, nuint buf_size);

    [LibraryImport(LibName)]
    internal static partial int llama_model_meta_count(IntPtr model);

    [LibraryImport(LibName)]
    internal static unsafe partial int llama_model_meta_key_by_index(
        IntPtr model, int i, byte* buf, nuint buf_size);

    [LibraryImport(LibName)]
    internal static unsafe partial int llama_model_meta_val_str_by_index(
        IntPtr model, int i, byte* buf, nuint buf_size);

    // ----- Context create / metadata / free -----

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_init_from_model(IntPtr model, llama_context_params params_);

    [LibraryImport(LibName)]
    internal static partial void llama_free(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial uint llama_n_ctx(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial uint llama_n_batch(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial uint llama_n_ubatch(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial uint llama_n_seq_max(IntPtr ctx);

    // ----- Capability queries -----

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_supports_gpu_offload();

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_supports_mmap();

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_supports_mlock();

    [LibraryImport(LibName)]
    internal static partial nuint llama_max_devices();

    // ----- Vocabulary (Phase 2) -----

    [LibraryImport(LibName)]
    internal static partial int llama_vocab_n_tokens(IntPtr vocab);

    // Special token getters. Return LLAMA_TOKEN_NULL (-1) if the model has no such token.
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_bos(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_eos(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_eot(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_sep(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_nl(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_pad(IntPtr vocab);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_vocab_is_eog(IntPtr vocab, int token);

    // ----- Vocab advanced (Tier 1 expansion) -----

    [LibraryImport(LibName)]
    internal static partial llama_vocab_type llama_vocab_type(IntPtr vocab);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_vocab_get_text(IntPtr vocab, int token);

    [LibraryImport(LibName)]
    internal static partial float llama_vocab_get_score(IntPtr vocab, int token);

    [LibraryImport(LibName)]
    internal static partial llama_token_attr llama_vocab_get_attr(IntPtr vocab, int token);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_vocab_is_control(IntPtr vocab, int token);

    [LibraryImport(LibName)]
    internal static partial int llama_vocab_mask(IntPtr vocab);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_vocab_get_add_bos(IntPtr vocab);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_vocab_get_add_eos(IntPtr vocab);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_vocab_get_add_sep(IntPtr vocab);

    // FIM (fill-in-the-middle) tokens for code-completion models.
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_fim_pre(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_fim_suf(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_fim_mid(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_fim_pad(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_fim_rep(IntPtr vocab);
    [LibraryImport(LibName)]
    internal static partial int llama_vocab_fim_sep(IntPtr vocab);

    // ----- Tokenization (Phase 2) -----
    //
    // Return: number of tokens on success, or a negative count indicating the
    // required buffer size on overflow (i.e. call again with a buffer of size
    // -return_value).

    [LibraryImport(LibName)]
    internal static unsafe partial int llama_tokenize(
        IntPtr vocab,
        byte* text, int text_len,
        int* tokens, int n_tokens_max,
        [MarshalAs(UnmanagedType.I1)] bool add_special,
        [MarshalAs(UnmanagedType.I1)] bool parse_special);

    // Return: bytes written (or needed, as negative) into `buf`. No null terminator.
    [LibraryImport(LibName)]
    internal static unsafe partial int llama_token_to_piece(
        IntPtr vocab,
        int token,
        byte* buf, int length,
        int lstrip,
        [MarshalAs(UnmanagedType.I1)] bool special);

    // ----- Chat templating (Phase 2) -----
    //
    // llama_model_chat_template returns a pointer into the model's metadata;
    // the string is owned by the model and must not be freed.
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr llama_model_chat_template(IntPtr model, string? name);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int llama_chat_apply_template(
        string tmpl,
        llama_chat_message* chat,
        nuint n_msg,
        [MarshalAs(UnmanagedType.I1)] bool add_ass,
        byte* buf,
        int length);

    // ----- Batch (Phase 3) -----
    //
    // llama_batch_get_one wraps an array of tokens as a single-sequence batch
    // without any allocation — the returned struct references the caller's
    // array directly. Caller must keep the array pinned for the duration of
    // llama_decode.

    [LibraryImport(LibName)]
    internal static unsafe partial llama_batch llama_batch_get_one(int* tokens, int n_tokens);

    [LibraryImport(LibName)]
    internal static partial llama_batch llama_batch_init(int n_tokens, int embd, int n_seq_max);

    [LibraryImport(LibName)]
    internal static partial void llama_batch_free(llama_batch batch);

    // ----- Decode / logits (Phase 3) -----
    //
    // llama_decode return codes:
    //   0  = success
    //   1  = could not find a KV slot (try a smaller batch or larger n_ctx)
    //   2  = aborted
    //   -1 = invalid input batch
    //   <-1 = fatal error
    [LibraryImport(LibName)]
    internal static partial int llama_decode(IntPtr ctx, llama_batch batch);

    // Runs the encoder path on a batch (for encoder-decoder models). Return
    // codes mirror llama_decode: 0 success, negative = error.
    [LibraryImport(LibName)]
    internal static partial int llama_encode(IntPtr ctx, llama_batch batch);

    // Returns a pointer into the context's internal embedding buffer —
    // valid until the next decode / encode call. Length is n_embd floats.
    [LibraryImport(LibName)]
    internal static unsafe partial float* llama_get_embeddings(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static unsafe partial float* llama_get_embeddings_ith(IntPtr ctx, int i);

    [LibraryImport(LibName)]
    internal static unsafe partial float* llama_get_embeddings_seq(IntPtr ctx, int seq_id);

    [LibraryImport(LibName)]
    internal static unsafe partial float* llama_get_logits_ith(IntPtr ctx, int i);

    [LibraryImport(LibName)]
    internal static unsafe partial float* llama_get_logits(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial void llama_set_n_threads(IntPtr ctx, int n_threads, int n_threads_batch);

    // ----- Sampler chain (Phase 3) -----
    //
    // Ownership rule: llama_sampler_chain_add takes ownership of the added
    // sub-sampler. Freeing the chain frees all contained sub-samplers. We
    // therefore SafeHandle-wrap only the chain; raw sub-sampler pointers
    // returned by llama_sampler_init_* are transferred into the chain and
    // must NOT be freed independently.

    [LibraryImport(LibName)]
    internal static partial llama_sampler_chain_params llama_sampler_chain_default_params();

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_chain_init(llama_sampler_chain_params params_);

    [LibraryImport(LibName)]
    internal static partial void llama_sampler_chain_add(IntPtr chain, IntPtr sub);

    [LibraryImport(LibName)]
    internal static partial void llama_sampler_free(IntPtr smpl);

    [LibraryImport(LibName)]
    internal static partial void llama_sampler_reset(IntPtr smpl);

    [LibraryImport(LibName)]
    internal static partial int llama_sampler_sample(IntPtr smpl, IntPtr ctx, int idx);

    [LibraryImport(LibName)]
    internal static partial void llama_sampler_accept(IntPtr smpl, int token);

    // Individual samplers — all return a newly-allocated llama_sampler* that
    // callers typically hand off to llama_sampler_chain_add.

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_greedy();

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_dist(uint seed);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_top_k(int k);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_top_p(float p, nuint min_keep);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_min_p(float p, nuint min_keep);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_typical(float p, nuint min_keep);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_temp(float t);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_temp_ext(float t, float delta, float exponent);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_penalties(
        int penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present);

    // ----- Advanced samplers (Tier 2) -----

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_xtc(float p, float t, nuint min_keep, uint seed);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_top_n_sigma(float n);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_mirostat(int n_vocab, uint seed, float tau, float eta, int m);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_mirostat_v2(uint seed, float tau, float eta);

    [LibraryImport(LibName)]
    internal static unsafe partial IntPtr llama_sampler_init_dry(
        IntPtr vocab,
        int n_ctx_train,
        float dry_multiplier, float dry_base,
        int dry_allowed_length, int dry_penalty_last_n,
        IntPtr* seq_breakers, nuint num_breakers);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_adaptive_p(float target, float decay, uint seed);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_init_infill(IntPtr vocab);

    [LibraryImport(LibName)]
    internal static unsafe partial IntPtr llama_sampler_init_logit_bias(
        int n_vocab, int n_logit_bias, llama_logit_bias* logit_bias);

    // ----- Grammar sampling (Tier 2) -----

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr llama_sampler_init_grammar(
        IntPtr vocab, string grammar_str, string grammar_root);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial IntPtr llama_sampler_init_grammar_lazy_patterns(
        IntPtr vocab,
        string grammar_str,
        string grammar_root,
        IntPtr* trigger_patterns, nuint num_trigger_patterns,
        int* trigger_tokens, nuint num_trigger_tokens);

    // ----- Sampler introspection (Tier 1 expansion) -----
    //
    // llama_sampler_chain_get returns a pointer INTO the chain; the pointer
    // is NOT owned by the caller and must not be freed. We don't expose it
    // directly as a managed type for that reason — sub-sampler introspection
    // goes through GetChainStageName(index) which calls chain_get + sampler_name
    // in sequence without handing back the raw pointer.
    //
    // llama_sampler_apply and llama_sampler_chain_remove remain deliberately
    // unbound:
    //   - apply needs llama_token_data_array mirroring (Tier-2 custom sampling)
    //   - chain_remove transfers ownership back to caller (free hazard)

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_name(IntPtr smpl);

    // Applies a sampler (single or chain) to a candidate array. Mutates in
    // place: .size may shrink (truncation samplers), logits may be rescaled
    // (temperature), masked to -inf (grammar / logit_bias), or reordered.
    // The .selected field is set by terminal samplers (dist/greedy/mirostat/
    // adaptive_p). Importantly: apply does NOT advance stateful samplers
    // (grammar position, penalty history, etc.) — that's what accept() does.
    // Safe to call multiple times without side effects on sampler state.
    [LibraryImport(LibName)]
    internal static unsafe partial void llama_sampler_apply(IntPtr smpl, llama_token_data_array* cur_p);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_clone(IntPtr smpl);

    [LibraryImport(LibName)]
    internal static partial int llama_sampler_chain_n(IntPtr chain);

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_sampler_chain_get(IntPtr chain, int i);

    [LibraryImport(LibName)]
    internal static partial uint llama_sampler_get_seed(IntPtr smpl);

    // ----- Perf readouts (Tier 1 expansion) -----
    //
    // The _print functions log to stderr via llama_log; we still bind them
    // for diagnostic use even though the managed API prefers returning the
    // struct for callers to render themselves.

    [LibraryImport(LibName)]
    internal static partial llama_perf_context_data llama_perf_context(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial void llama_perf_context_print(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial void llama_perf_context_reset(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial llama_perf_sampler_data llama_perf_sampler(IntPtr chain);

    [LibraryImport(LibName)]
    internal static partial void llama_perf_sampler_print(IntPtr chain);

    [LibraryImport(LibName)]
    internal static partial void llama_perf_sampler_reset(IntPtr chain);

    // ----- Memory / KV cache (Phase 4) -----
    //
    // Note: in this pinned version the KV cache is reached via an opaque
    // llama_memory_t pointer (obtained from llama_get_memory(ctx)) rather than
    // the old llama_kv_self_* family. Functionally equivalent; names changed.

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_get_memory(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial void llama_memory_clear(IntPtr mem, [MarshalAs(UnmanagedType.I1)] bool data);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_memory_seq_rm(IntPtr mem, int seq_id, int p0, int p1);

    [LibraryImport(LibName)]
    internal static partial void llama_memory_seq_cp(IntPtr mem, int seq_id_src, int seq_id_dst, int p0, int p1);

    [LibraryImport(LibName)]
    internal static partial void llama_memory_seq_keep(IntPtr mem, int seq_id);

    [LibraryImport(LibName)]
    internal static partial int llama_memory_seq_pos_min(IntPtr mem, int seq_id);

    [LibraryImport(LibName)]
    internal static partial int llama_memory_seq_pos_max(IntPtr mem, int seq_id);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_memory_can_shift(IntPtr mem);

    // ----- Extra memory ops (Tier 1 expansion) -----

    // Add `delta` to every position in [p0, p1) for the given sequence.
    // Used by sliding-window truncation: remove [0, k), then seq_add(k, inf, -k)
    // to renumber surviving tokens back down to start at 0.
    [LibraryImport(LibName)]
    internal static partial void llama_memory_seq_add(IntPtr mem, int seq_id, int p0, int p1, int delta);

    // Divide every position in [p0, p1) by `d`. Rarely used; enables aggressive
    // compression schemes that re-map position ids onto a coarser grid.
    [LibraryImport(LibName)]
    internal static partial void llama_memory_seq_div(IntPtr mem, int seq_id, int p0, int p1, int d);

    // Prints a per-device memory breakdown via llama.cpp's log sink. Diagnostic.
    [LibraryImport(LibName)]
    internal static partial void llama_memory_breakdown_print(IntPtr ctx);

    // ----- Capability / misc (Tier 1 expansion) -----

    [LibraryImport(LibName)]
    internal static partial void llama_numa_init(ggml_numa_strategy strategy);

    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool llama_supports_rpc();

    [LibraryImport(LibName)]
    internal static partial nuint llama_max_parallel_sequences();

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_print_system_info();

    [LibraryImport(LibName)]
    internal static partial IntPtr llama_flash_attn_type_name(llama_flash_attn_type type);

    [LibraryImport(LibName)]
    internal static partial uint llama_n_ctx_seq(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial int llama_n_threads(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial int llama_n_threads_batch(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial llama_pooling_type llama_pooling_type(IntPtr ctx);

    [LibraryImport(LibName)]
    internal static partial void llama_set_embeddings(IntPtr ctx, [MarshalAs(UnmanagedType.I1)] bool embeddings);

    [LibraryImport(LibName)]
    internal static partial void llama_set_causal_attn(IntPtr ctx, [MarshalAs(UnmanagedType.I1)] bool causal_attn);

    [LibraryImport(LibName)]
    internal static partial void llama_set_warmup(IntPtr ctx, [MarshalAs(UnmanagedType.I1)] bool warmup);

    [LibraryImport(LibName)]
    internal static partial void llama_synchronize(IntPtr ctx);

    // Proper bulk detokenize with special-token handling.
    [LibraryImport(LibName)]
    internal static unsafe partial int llama_detokenize(
        IntPtr vocab,
        int* tokens, int n_tokens,
        byte* text, int text_len_max,
        [MarshalAs(UnmanagedType.I1)] bool remove_special,
        [MarshalAs(UnmanagedType.I1)] bool unparse_special);

    [LibraryImport(LibName)]
    internal static unsafe partial int llama_chat_builtin_templates(IntPtr* output, nuint len);

    // ----- Multimodal (libmtmd) -----
    //
    // mtmd is a sibling library to libllama (libmtmd.so / mtmd.dll /
    // libmtmd.dylib). The NativeLibraryResolver knows how to map "mtmd" to
    // the correct on-disk filename per OS. C names mirrored verbatim per
    // CLAUDE.md.

    [LibraryImport(LibMtmd)]
    internal static partial IntPtr mtmd_default_marker();

    [LibraryImport(LibMtmd)]
    internal static partial mtmd_context_params mtmd_context_params_default();

    [LibraryImport(LibMtmd, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr mtmd_init_from_file(
        string mmproj_fname,
        IntPtr text_model,
        mtmd_context_params ctx_params);

    [LibraryImport(LibMtmd)]
    internal static partial void mtmd_free(IntPtr ctx);

    [LibraryImport(LibMtmd)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool mtmd_decode_use_non_causal(IntPtr ctx);

    [LibraryImport(LibMtmd)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool mtmd_decode_use_mrope(IntPtr ctx);

    [LibraryImport(LibMtmd)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool mtmd_support_vision(IntPtr ctx);

    [LibraryImport(LibMtmd)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool mtmd_support_audio(IntPtr ctx);

    [LibraryImport(LibMtmd)]
    internal static partial int mtmd_get_audio_sample_rate(IntPtr ctx);

    // Bitmap
    [LibraryImport(LibMtmd)]
    internal static unsafe partial IntPtr mtmd_bitmap_init(uint nx, uint ny, byte* data);

    [LibraryImport(LibMtmd)]
    internal static unsafe partial IntPtr mtmd_bitmap_init_from_audio(nuint n_samples, float* data);

    [LibraryImport(LibMtmd)]
    internal static partial uint mtmd_bitmap_get_nx(IntPtr bitmap);

    [LibraryImport(LibMtmd)]
    internal static partial uint mtmd_bitmap_get_ny(IntPtr bitmap);

    [LibraryImport(LibMtmd)]
    internal static partial IntPtr mtmd_bitmap_get_data(IntPtr bitmap);

    [LibraryImport(LibMtmd)]
    internal static partial nuint mtmd_bitmap_get_n_bytes(IntPtr bitmap);

    [LibraryImport(LibMtmd)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static partial bool mtmd_bitmap_is_audio(IntPtr bitmap);

    [LibraryImport(LibMtmd)]
    internal static partial void mtmd_bitmap_free(IntPtr bitmap);

    [LibraryImport(LibMtmd)]
    internal static partial IntPtr mtmd_bitmap_get_id(IntPtr bitmap);

    [LibraryImport(LibMtmd, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mtmd_bitmap_set_id(IntPtr bitmap, string id);

    // Input chunks
    [LibraryImport(LibMtmd)]
    internal static partial IntPtr mtmd_input_chunks_init();

    [LibraryImport(LibMtmd)]
    internal static partial nuint mtmd_input_chunks_size(IntPtr chunks);

    [LibraryImport(LibMtmd)]
    internal static partial IntPtr mtmd_input_chunks_get(IntPtr chunks, nuint idx);

    [LibraryImport(LibMtmd)]
    internal static partial void mtmd_input_chunks_free(IntPtr chunks);

    [LibraryImport(LibMtmd)]
    internal static partial mtmd_input_chunk_type mtmd_input_chunk_get_type(IntPtr chunk);

    [LibraryImport(LibMtmd)]
    internal static partial nuint mtmd_input_chunk_get_n_tokens(IntPtr chunk);

    [LibraryImport(LibMtmd)]
    internal static partial int mtmd_input_chunk_get_n_pos(IntPtr chunk);

    [LibraryImport(LibMtmd)]
    internal static partial IntPtr mtmd_input_chunk_get_id(IntPtr chunk);

    // Tokenize: output chunks get filled from (text + image markers + bitmaps).
    // Status: 0 ok, 1 bitmap count != marker count, 2 image preprocessing failure.
    [LibraryImport(LibMtmd)]
    internal static unsafe partial int mtmd_tokenize(
        IntPtr ctx,
        IntPtr output,
        in mtmd_input_text text,
        IntPtr* bitmaps,
        nuint n_bitmaps);

    [LibraryImport(LibMtmd)]
    internal static partial void mtmd_log_set(IntPtr log_callback, IntPtr user_data);

    // ----- Multimodal helper (libmtmd) -----

    [LibraryImport(LibMtmd)]
    internal static partial void mtmd_helper_log_set(IntPtr log_callback, IntPtr user_data);

    [LibraryImport(LibMtmd, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr mtmd_helper_bitmap_init_from_file(IntPtr ctx, string fname);

    [LibraryImport(LibMtmd)]
    internal static unsafe partial IntPtr mtmd_helper_bitmap_init_from_buf(
        IntPtr ctx, byte* buf, nuint len);

    [LibraryImport(LibMtmd)]
    internal static partial nuint mtmd_helper_get_n_tokens(IntPtr chunks);

    [LibraryImport(LibMtmd)]
    internal static partial int mtmd_helper_get_n_pos(IntPtr chunks);

    // Runs mtmd_encode + llama_decode across every chunk. Returns 0 on success,
    // forwards the first nonzero status otherwise. NOT thread-safe (documented
    // on the header). new_n_past receives the post-prefill position.
    [LibraryImport(LibMtmd)]
    internal static partial int mtmd_helper_eval_chunks(
        IntPtr ctx,
        IntPtr lctx,
        IntPtr chunks,
        int n_past,
        int seq_id,
        int n_batch,
        [MarshalAs(UnmanagedType.I1)] bool logits_last,
        out int new_n_past);
}
