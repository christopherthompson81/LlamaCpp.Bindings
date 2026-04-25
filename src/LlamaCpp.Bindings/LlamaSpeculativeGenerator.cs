using System.Runtime.CompilerServices;
using System.Text;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Two-model speculative decoding. A small, fast <em>draft</em> model proposes
/// <see cref="DraftLookahead"/> tokens ahead; the large <em>main</em> model
/// verifies them in a single parallel decode. When the draft agrees with the
/// main's own pick at a position, that token is accepted and we've effectively
/// saved a full main-model decode step. On first disagreement the main's token
/// replaces the draft's and speculation restarts from there.
/// </summary>
/// <remarks>
/// <para>Typical speedup is 2–5× when the draft distribution closely matches
/// the main's. Choose a draft that shares the main's tokenizer family and has
/// been trained on similar data — a mismatched draft can be slower than plain
/// single-model decoding because every rejection still costs one draft decode
/// plus one (wasted) main batch slot.</para>
///
/// <para><b>Vocabulary compatibility (v1):</b> both contexts' models must share
/// the same tokenizer — we tokenize the prompt once with the main model's
/// vocabulary and feed the same IDs to the draft. Cross-vocab speculation
/// (e.g. Llama draft with Qwen main) would require retokenization round-trips
/// and is out of scope here.</para>
///
/// <para><b>VRAM:</b> two models loaded simultaneously ≈ 1.5–2× the single-
/// model footprint. Pair a large main with a small draft (e.g. Qwen3-14B +
/// Qwen3-0.6B) so the draft cost is negligible.</para>
///
/// <para><b>Cancellation:</b> a mid-decode cancel may leave both contexts
/// with partially-decoded speculative tokens. After a cancelled generation,
/// call <see cref="LlamaContext.ClearKvCache"/> on both contexts before
/// reusing them for unrelated work.</para>
/// </remarks>
public sealed class LlamaSpeculativeGenerator : IDisposable
{
    private readonly LlamaContext _main;
    private readonly LlamaContext _draft;
    private readonly LlamaSampler _mainSampler;
    private readonly LlamaSampler _draftSampler;

    private llama_batch _verifyBatch;
    private bool _disposed;

    /// <summary>Maximum draft lookahead per speculation round.</summary>
    public int DraftLookahead { get; }

    /// <summary>Running acceptance counters. Updated across <c>GenerateAsync</c> calls; call <see cref="ResetStats"/> to zero them.</summary>
    public SpeculativeStats Stats { get; private set; }

    /// <summary>Reason the most recent generation loop stopped.</summary>
    public LlamaStopReason LastStopReason { get; private set; } = LlamaStopReason.None;

    /// <summary>
    /// Wire up a speculative decoder. Both contexts must have been created from
    /// models with compatible tokenizers (<see cref="LlamaVocab.TokenCount"/>
    /// equal and special tokens matching). Ownership of the contexts and
    /// samplers remains with the caller — <c>Dispose</c> only releases the
    /// verification batch allocated by this class.
    /// </summary>
    public LlamaSpeculativeGenerator(
        LlamaContext main,
        LlamaContext draft,
        LlamaSampler mainSampler,
        LlamaSampler draftSampler,
        int draftLookahead = 5)
    {
        ArgumentNullException.ThrowIfNull(main);
        ArgumentNullException.ThrowIfNull(draft);
        ArgumentNullException.ThrowIfNull(mainSampler);
        ArgumentNullException.ThrowIfNull(draftSampler);
        if (ReferenceEquals(main, draft))
        {
            throw new ArgumentException(
                "main and draft must be distinct contexts — a self-speculation loop " +
                "would decode each token twice without any throughput benefit.",
                nameof(draft));
        }
        if (draftLookahead < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(draftLookahead),
                "Draft lookahead must be at least 1.");
        }

        EnsureVocabsCompatible(main.Model.Vocab, draft.Model.Vocab);

        _main = main;
        _draft = draft;
        _mainSampler = mainSampler;
        _draftSampler = draftSampler;
        DraftLookahead = draftLookahead;

        // Single allocation reused every speculation round. Size = lookahead + 1
        // because the verification batch always prepends id_last (the token that
        // drove the last draft round) so the main model can produce logits at
        // every position it needs to compare against.
        _verifyBatch = NativeMethods.llama_batch_init(draftLookahead + 1, embd: 0, n_seq_max: 1);
    }

    /// <summary>Zero the acceptance counters.</summary>
    public void ResetStats() => Stats = default;

    // Tokenizer compatibility gate, mirroring common_speculative_are_compatible
    // in llama.cpp (see common/speculative.cpp). Speculative decoding relies on
    // draft token ids being directly usable by the main — that only holds if
    // the two vocabularies agree on token type, special-token ids, and the
    // textual content of the low-index tokens. Relaxing any of these produces
    // silent divergence between the speculative stream and plain decoding.
    private const int MaxVocabSizeDifference = 128;
    private const int VocabTextCompareStart  = 5;   // skip special-token slots
    private const int VocabTextCompareEnd    = 256; // enough to catch family mismatches

    private static void EnsureVocabsCompatible(LlamaVocab main, LlamaVocab draft)
    {
        if (main.VocabType != draft.VocabType)
        {
            throw new ArgumentException(
                $"Draft vocab type ({draft.VocabType}) differs from main ({main.VocabType}). " +
                "Speculative decoding requires identical tokenizer families (SPM / BPE / etc.).",
                nameof(draft));
        }

        if (main.AddsBosAutomatically != draft.AddsBosAutomatically ||
            main.AddsEosAutomatically != draft.AddsEosAutomatically ||
            main.Bos != draft.Bos ||
            main.Eos != draft.Eos)
        {
            throw new ArgumentException(
                "Main and draft vocabularies disagree on special tokens " +
                $"(BOS main={Describe(main.Bos)} draft={Describe(draft.Bos)}, " +
                $"EOS main={Describe(main.Eos)} draft={Describe(draft.Eos)}, " +
                $"addBos main={main.AddsBosAutomatically} draft={draft.AddsBosAutomatically}, " +
                $"addEos main={main.AddsEosAutomatically} draft={draft.AddsEosAutomatically}). " +
                "Speculative decoding would silently emit different tokens than plain decoding.",
                nameof(draft));
        }

        int diff = Math.Abs(main.TokenCount - draft.TokenCount);
        if (diff > MaxVocabSizeDifference)
        {
            throw new ArgumentException(
                $"Main vocab ({main.TokenCount} tokens) and draft vocab ({draft.TokenCount} tokens) " +
                $"differ by {diff} — more than the {MaxVocabSizeDifference}-token tolerance that " +
                "llama.cpp uses for speculative decoding. Pick a draft from the same model family.",
                nameof(draft));
        }

        int end = Math.Min(Math.Min(main.TokenCount, draft.TokenCount), VocabTextCompareEnd);
        for (int i = VocabTextCompareStart; i < end; i++)
        {
            var mainText  = main.GetTokenText(i);
            var draftText = draft.GetTokenText(i);
            if (!string.Equals(mainText, draftText, StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Main and draft vocabularies disagree at token id {i}: " +
                    $"main='{mainText ?? "<null>"}', draft='{draftText ?? "<null>"}'. " +
                    "Even when token counts match, a draft trained with a different tokenizer " +
                    "would emit ids whose string meaning diverges in the main — correctness " +
                    "would be silently broken. Pair with a draft from the same family.",
                    nameof(draft));
            }
        }

        static string Describe(int? token) => token?.ToString() ?? "<none>";
    }

    /// <summary>
    /// Tokenize <paramref name="prompt"/> using the main model's vocabulary,
    /// ingest it into both contexts, then stream generated text pieces until
    /// EOG, the <paramref name="maxTokens"/> cap, or cancellation.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        int maxTokens = 512,
        bool addSpecial = false,
        bool parseSpecial = true,
        bool renderSpecialPieces = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prompt);
        if (maxTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTokens));

        var promptTokens = _main.Model.Vocab.Tokenize(prompt, addSpecial, parseSpecial);

        await foreach (var piece in GenerateAsync(
                                        promptTokens,
                                        maxTokens: maxTokens,
                                        renderSpecialPieces: renderSpecialPieces,
                                        cancellationToken: cancellationToken)
                                    .ConfigureAwait(false))
        {
            yield return piece;
        }
    }

    /// <summary>
    /// Pre-tokenized variant. The caller is responsible for ensuring the
    /// tokens were produced by a vocab compatible with both contexts.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        IReadOnlyList<int> promptTokens,
        int maxTokens = 512,
        bool renderSpecialPieces = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(promptTokens);
        if (promptTokens.Count < 1)
        {
            throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokens));
        }
        if (maxTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTokens));

        LastStopReason = LlamaStopReason.None;

        var vocab = _main.Model.Vocab;

        // 1) Ingest the prompt.
        //    Main decodes prompt[..^1]; the last token rides the first
        //    verification batch so the main produces logits at every position
        //    it needs. The draft decodes the full prompt (including the last
        //    token) so its logits are already at "next-token after id_last"
        //    when we enter the speculation loop.
        var promptArray = new int[promptTokens.Count];
        for (int i = 0; i < promptArray.Length; i++) promptArray[i] = promptTokens[i];

        int idLast = promptArray[^1];
        int nextPos = promptArray.Length - 1;

        cancellationToken.ThrowIfCancellationRequested();
        await Task.Run(() =>
        {
            if (promptArray.Length > 1)
            {
                DecodeRange(_main, promptArray, 0, promptArray.Length - 1);
            }
            DecodeRange(_draft, promptArray, 0, promptArray.Length);
        }, cancellationToken).ConfigureAwait(false);

        // Prime samplers with the full prompt so history-aware stages
        // (penalties, DRY) treat prompt-word repetition as repetition. This
        // mirrors LlamaGenerator's prompt-priming path.
        foreach (var t in promptArray) _mainSampler.Accept(t);
        foreach (var t in promptArray) _draftSampler.Accept(t);

        // 2) Speculation loop.
        var decoder = Encoding.UTF8.GetDecoder();
        var charBuf = new char[64];
        var draftBuf = new int[DraftLookahead];
        int emitted = 0;

        while (emitted < maxTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // 2a) Draft generates up to DraftLookahead tokens. Stops early if
            //     the draft itself emits EOG — the main will see that EOG and
            //     either agree (we terminate cleanly) or disagree (we replace
            //     it with the main's correction and keep going).
            int driftCount = 0;
            bool draftHitEog = false;
            for (int i = 0; i < DraftLookahead; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int t = await Task.Run(() => _draftSampler.Sample(_draft), cancellationToken).ConfigureAwait(false);

                if (vocab.IsEndOfGeneration(t))
                {
                    draftHitEog = true;
                    break;
                }

                draftBuf[driftCount++] = t;
                await Task.Run(() => DecodeSingle(_draft, t), cancellationToken).ConfigureAwait(false);
            }

            // 2b) Build the verification batch: [id_last, draft[0..driftCount-1]]
            //     at positions [nextPos, nextPos+1, ..., nextPos+driftCount],
            //     every slot with logits=true so we can sample at every
            //     position after the main's decode.
            int verifyCount = driftCount + 1;
            BuildVerifyBatch(idLast, draftBuf.AsSpan(0, driftCount), nextPos);

            cancellationToken.ThrowIfCancellationRequested();
            await Task.Run(() => DecodeVerify(), cancellationToken).ConfigureAwait(false);

            // 2c) Walk the verification positions. Position i=0 corresponds
            //     to the post-id_last slot, predicting what should come after
            //     id_last (compare to draft[0]). Position i>0 predicts what
            //     follows draft[i-1] (compare to draft[i] when i<driftCount,
            //     or is the bonus beyond the draft when i==driftCount).
            int acceptedThisRound = 0;
            bool mainHitEog = false;
            int correctionToken = 0;

            for (int i = 0; i < verifyCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int sampled = await Task.Run(() => _mainSampler.Sample(_main, i), cancellationToken).ConfigureAwait(false);

                if (vocab.IsEndOfGeneration(sampled))
                {
                    mainHitEog = true;
                    correctionToken = sampled;
                    break;
                }

                if (i < driftCount && sampled == draftBuf[i])
                {
                    // Main would have picked the same token — accept the draft
                    // guess and move on. Don't double-accept into the sampler:
                    // Sample() already called llama_sampler_accept internally.
                    acceptedThisRound++;
                    continue;
                }

                // Either a mismatch (i < driftCount and sampled != draft[i])
                // or the all-accepted bonus slot (i == driftCount). Either way,
                // sampled becomes the next committed token.
                correctionToken = sampled;
                break;
            }

            // 2d) Emit tokens: driftCount accepted drafts (up to acceptedThisRound)
            //     followed by the correction/bonus/EOG token.
            for (int i = 0; i < acceptedThisRound && emitted < maxTokens; i++)
            {
                EmitPiece(draftBuf[i], decoder, ref charBuf, renderSpecialPieces, out var piece);
                if (piece.Length > 0) yield return piece;
                emitted++;
            }

            if (mainHitEog)
            {
                FlushDecoder(decoder, ref charBuf, out var tail);
                if (tail.Length > 0) yield return tail;
                TrimToAccepted(nextPos, acceptedThisRound);
                Stats = new SpeculativeStats(Stats.TotalDrafted + driftCount, Stats.TotalAccepted + acceptedThisRound);
                LastStopReason = LlamaStopReason.EndOfGeneration;
                yield break;
            }

            if (emitted < maxTokens)
            {
                EmitPiece(correctionToken, decoder, ref charBuf, renderSpecialPieces, out var piece);
                if (piece.Length > 0) yield return piece;
                emitted++;
            }

            Stats = new SpeculativeStats(Stats.TotalDrafted + driftCount, Stats.TotalAccepted + acceptedThisRound);

            // 2e) Roll both contexts' KV caches back to the committed prefix.
            //     Main currently holds nextPos + verifyCount positions; keep
            //     nextPos + acceptedThisRound + 1 (id_last + accepted drafts)
            //     and discard anything past that. The correction/bonus token
            //     is NOT in main's KV yet — it rides the next verification
            //     batch as id_last.
            TrimToAccepted(nextPos, acceptedThisRound);

            // Draft KV needs the same prefix shape. It holds
            // (prompt-length + prevDraftCount + driftCount) positions across
            // the run; accepting `acceptedThisRound` of the draft round keeps
            // positions up to (previous end + acceptedThisRound). The
            // correction then needs to be ingested so the next round's first
            // sample sees it.
            int draftKeepTo = nextPos + acceptedThisRound + 1;
            await Task.Run(() =>
            {
                // Keep [0, draftKeepTo) tokens — draft KV currently goes up to
                // (nextPos + driftCount) inclusive (positions 0..nextPos+driftCount).
                // Trim past draftKeepTo, i.e. drop positions >= draftKeepTo.
                _draft.RemoveSequenceRange(sequenceId: 0, fromPosition: draftKeepTo, toPosition: -1);
                // Feed the correction into draft so its logits are ready for
                // the next round's first draft sample.
                DecodeSingleAt(_draft, correctionToken, draftKeepTo);
            }, cancellationToken).ConfigureAwait(false);

            // Update bookkeeping for next iteration.
            idLast = correctionToken;
            nextPos = draftKeepTo;

            // Draft EOG that was agreed upon by the main would have been
            // caught as an EOG on the main side. If only the draft hit EOG
            // but the main disagreed, we replaced with the correction —
            // continue speculating. The draft is positioned at the
            // correction, so next round starts clean.
            _ = draftHitEog;

            // If the main's bonus also hit max tokens, stop.
            if (emitted >= maxTokens) break;
        }

        FlushDecoder(decoder, ref charBuf, out var trailing);
        if (trailing.Length > 0) yield return trailing;
        LastStopReason = LlamaStopReason.MaxTokens;
    }

    private void TrimToAccepted(int nextPos, int acceptedThisRound)
    {
        // Main holds positions 0..(nextPos + driftCount); we keep through
        // position (nextPos + acceptedThisRound) — that's id_last at nextPos
        // plus the accepted drafts. Trim removes [keepTo, end).
        int keepTo = nextPos + acceptedThisRound + 1;
        _main.RemoveSequenceRange(sequenceId: 0, fromPosition: keepTo, toPosition: -1);
    }

    private unsafe void BuildVerifyBatch(int idLast, ReadOnlySpan<int> drafts, int nextPos)
    {
        // The batch arrays were allocated by llama_batch_init for up to
        // (DraftLookahead + 1) tokens; we just overwrite them each round.
        var batch = _verifyBatch;
        int count = drafts.Length + 1;
        batch.n_tokens = count;

        var tokens   = (int*)batch.token;
        var pos      = (int*)batch.pos;
        var nSeqId   = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;

        tokens[0]  = idLast;
        pos[0]     = nextPos;
        nSeqId[0]  = 1;
        seqIdArr[0][0] = 0;
        logits[0]  = 1;

        for (int i = 0; i < drafts.Length; i++)
        {
            int slot = i + 1;
            tokens[slot]  = drafts[i];
            pos[slot]     = nextPos + slot;
            nSeqId[slot]  = 1;
            seqIdArr[slot][0] = 0;
            logits[slot]  = 1;
        }

        _verifyBatch = batch;
    }

    private void DecodeVerify()
    {
        var rc = NativeMethods.llama_decode(_main.Handle.DangerousHandle, _verifyBatch);
        if (rc != 0)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_decode), rc,
                $"Speculative verification decode failed with status {rc}. " +
                (rc == 1 ? "Code 1 = no KV slot; reduce draft lookahead or increase context." : ""));
        }
    }

    private unsafe void DecodeRange(LlamaContext ctx, int[] tokens, int start, int count)
    {
        if (count <= 0) return;
        int cap = Math.Max(1, ctx.LogicalBatchSize);
        int offset = start;
        int end = start + count;
        while (offset < end)
        {
            int take = Math.Min(cap, end - offset);
            fixed (int* tokPtr = &tokens[offset])
            {
                var batch = NativeMethods.llama_batch_get_one(tokPtr, take);
                var rc = NativeMethods.llama_decode(ctx.Handle.DangerousHandle, batch);
                if (rc != 0)
                {
                    throw new LlamaException(
                        nameof(NativeMethods.llama_decode), rc,
                        $"llama_decode returned {rc} during speculative prompt ingest " +
                        $"(offset {offset}/{end}, take {take}).");
                }
            }
            offset += take;
        }
    }

    private static unsafe void DecodeSingle(LlamaContext ctx, int token)
    {
        // llama_batch_get_one auto-positions from the context's KV tail.
        var toks = stackalloc int[1]; toks[0] = token;
        var batch = NativeMethods.llama_batch_get_one(toks, 1);
        var rc = NativeMethods.llama_decode(ctx.Handle.DangerousHandle, batch);
        if (rc != 0)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_decode), rc,
                $"llama_decode returned {rc} during speculative single-token step.");
        }
    }

    private static unsafe void DecodeSingleAt(LlamaContext ctx, int token, int position)
    {
        // We need an explicit position because the draft context's KV tail
        // may have been rolled back via RemoveSequenceRange — llama.cpp's
        // auto-positioning reads the memory head, so in the common case
        // batch_get_one would still do the right thing; we use a
        // single-slot pre-allocated batch for determinism and clarity.
        int* toks = stackalloc int[1]; toks[0] = token;
        int* pos  = stackalloc int[1]; pos[0]  = position;
        int* nSeq = stackalloc int[1]; nSeq[0] = 1;
        int  seqZero = 0;
        int* seqPtr  = &seqZero;
        int** seqArr = &seqPtr;
        sbyte* logits = stackalloc sbyte[1]; logits[0] = 1;

        var batch = new llama_batch
        {
            n_tokens = 1,
            token    = (IntPtr)toks,
            embd     = IntPtr.Zero,
            pos      = (IntPtr)pos,
            n_seq_id = (IntPtr)nSeq,
            seq_id   = (IntPtr)seqArr,
            logits   = (IntPtr)logits,
        };
        var rc = NativeMethods.llama_decode(ctx.Handle.DangerousHandle, batch);
        if (rc != 0)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_decode), rc,
                $"llama_decode returned {rc} while re-seeding draft after speculative rollback.");
        }
    }

    private void EmitPiece(int token, Decoder decoder, ref char[] charBuf, bool renderSpecial, out string piece)
    {
        var bytes = GetPieceBytes(token, renderSpecial);
        if (bytes.Length == 0)
        {
            piece = string.Empty;
            return;
        }
        int charCount = decoder.GetChars(bytes, charBuf, flush: false);
        while (charCount == 0 && bytes.Length > charBuf.Length)
        {
            Array.Resize(ref charBuf, bytes.Length * 2);
            charCount = decoder.GetChars(bytes, charBuf, flush: false);
        }
        piece = charCount > 0 ? new string(charBuf, 0, charCount) : string.Empty;
    }

    private static void FlushDecoder(Decoder decoder, ref char[] charBuf, out string tail)
    {
        int final = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
        tail = final > 0 ? new string(charBuf, 0, final) : string.Empty;
    }

    private unsafe byte[] GetPieceBytes(int token, bool renderSpecial)
    {
        const int StackBuf = 64;
        Span<byte> buf = stackalloc byte[StackBuf];
        int n = CallTokenToPiece(token, buf, renderSpecial);

        if (n < 0)
        {
            var heap = new byte[-n];
            n = CallTokenToPiece(token, heap, renderSpecial);
            if (n < 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_token_to_piece), n,
                    $"llama_token_to_piece failed after retry for token {token}.");
            }
            var trimmed = new byte[n];
            Array.Copy(heap, trimmed, n);
            return trimmed;
        }

        if (n == 0) return [];
        var result = new byte[n];
        buf[..n].CopyTo(result);
        return result;
    }

    private unsafe int CallTokenToPiece(int token, Span<byte> buf, bool renderSpecial)
    {
        fixed (byte* bufPtr = buf)
        {
            return NativeMethods.llama_token_to_piece(
                _main.Model.Vocab.Handle, token, bufPtr, buf.Length, lstrip: 0, special: renderSpecial);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        NativeMethods.llama_batch_free(_verifyBatch);
        _verifyBatch = default;
    }
}

/// <summary>
/// Running counters for a <see cref="LlamaSpeculativeGenerator"/>. A draft
/// token is <em>drafted</em> every time the draft model proposes it and
/// <em>accepted</em> when the main model agrees with that proposal at its
/// verification pass. The bonus token sampled after full acceptance is not
/// counted as drafted (the draft didn't propose it) — so the ratio measures
/// pure draft-quality rather than overall throughput.
/// </summary>
public readonly record struct SpeculativeStats(int TotalDrafted, int TotalAccepted)
{
    /// <summary>Fraction of drafted tokens that the main model accepted. 0 when nothing has been drafted yet.</summary>
    public double AcceptanceRate => TotalDrafted == 0 ? 0.0 : (double)TotalAccepted / TotalDrafted;
}
