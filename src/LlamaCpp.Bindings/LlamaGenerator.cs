using System.Runtime.CompilerServices;
using System.Text;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// The streaming decode + sample loop. Takes a prompt (as tokens or as text
/// to be tokenized), feeds it to the context, and yields decoded text pieces
/// as tokens are produced.
/// </summary>
/// <remarks>
/// A <see cref="LlamaGenerator"/> wraps a single (<see cref="LlamaContext"/>,
/// <see cref="LlamaSampler"/>) pair. The generator does not own either — the
/// caller disposes them. This is deliberate: one model / context / sampler
/// often serves many generation turns, and coupling their lifetimes to a
/// single generation call would force wasteful re-loads.
/// </remarks>
public sealed class LlamaGenerator
{
    private readonly LlamaContext _context;
    private readonly LlamaSampler _sampler;
    private readonly int _sequenceId;

    /// <summary>
    /// Build a generator bound to <paramref name="context"/>'s implicit
    /// sequence 0. This is the original single-conversation ctor and the
    /// default when the context was created with
    /// <see cref="LlamaContextParameters.MaxSequenceCount"/> = 1.
    /// </summary>
    public LlamaGenerator(LlamaContext context, LlamaSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(sampler);
        _context = context;
        _sampler = sampler;
        _sequenceId = 0;
    }

    /// <summary>
    /// Build a generator bound to a specific <see cref="LlamaSession"/>.
    /// Tokens decoded, sampled, and yielded by this generator live in the
    /// session's dedicated sequence slot — concurrent sessions on the same
    /// context can run this ctor's output side-by-side without their
    /// histories colliding.
    /// </summary>
    public LlamaGenerator(LlamaSession session, LlamaSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(sampler);
        _context = session.Context;
        _sampler = sampler;
        _sequenceId = session.SequenceId;
    }

    /// <summary>
    /// Why the last call to a <c>*GenerateAsync</c> / <c>*StreamFromCurrentStateAsync</c>
    /// stopped. Set to <see cref="LlamaStopReason.None"/> when a new call
    /// starts, then updated just before the final <c>yield break</c> /
    /// normal loop exit. Cancellation throws through the iterator before
    /// we get a chance to annotate it, so callers that observe a
    /// cancellation should treat this as <see cref="LlamaStopReason.Cancelled"/>
    /// regardless of what's recorded.
    /// </summary>
    public LlamaStopReason LastStopReason { get; private set; } = LlamaStopReason.None;

    /// <summary>
    /// Tokenize <paramref name="prompt"/>, process it, then stream generated
    /// text pieces until the model emits an end-of-generation token or the
    /// cancellation token fires.
    /// </summary>
    /// <param name="prompt">
    /// Prompt text. If the host has already applied a chat template, pass
    /// <paramref name="parseSpecial"/> = <c>true</c> so tokenizer recognises
    /// <c>&lt;|im_start|&gt;</c>-style markers as single tokens.
    /// </param>
    /// <param name="maxTokens">
    /// Hard cap on emitted tokens (safety net, not a desired length). When
    /// reached, the generator returns normally without throwing.
    /// </param>
    /// <param name="addSpecial">Forward to <c>Tokenize</c> for prompt-ingest.</param>
    /// <param name="parseSpecial">Forward to <c>Tokenize</c> for prompt-ingest.</param>
    /// <param name="renderSpecialPieces">
    /// If true, detokenized pieces include special-token text (useful for
    /// debugging). Default false — the human-visible stream.
    /// </param>
    /// <param name="cancellationToken">Cooperative cancellation.</param>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        int maxTokens = 512,
        bool addSpecial = false,
        bool parseSpecial = true,
        bool renderSpecialPieces = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (maxTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTokens));

        var vocab = _context.Model.Vocab;
        var promptTokens = vocab.Tokenize(prompt, addSpecial, parseSpecial);

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
    /// Stream generation from the context's current state, skipping the prompt
    /// prefill + sampler priming. Use this when another code path (e.g.
    /// <see cref="MtmdContext.EvalPromptAsync"/>) has already populated the KV
    /// cache and left the last-position logits ready for sampling.
    /// </summary>
    /// <remarks>
    /// No prompt tokens are fed to the sampler, so history-aware stages
    /// (repetition penalty, presence penalty) start empty for this turn.
    /// Matches mtmd-cli.cpp's behavior — the prompt contributes to the KV
    /// cache but not to sampler history.
    /// </remarks>
    public async IAsyncEnumerable<string> StreamFromCurrentStateAsync(
        int maxTokens = 512,
        bool renderSpecialPieces = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (maxTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTokens));

        var vocab = _context.Model.Vocab;
        var decoder = Encoding.UTF8.GetDecoder();
        var charBuf = new char[64];
        int emitted = 0;
        LastStopReason = LlamaStopReason.None;

        // Sample/decode pairs serialize through the context's decode lock so
        // a second session driving the same context can't clobber our
        // last-position logits between our decode and our sample. See
        // LlamaContext.WithDecodeLockAsync.
        int? prev = null;
        while (emitted < maxTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int nextToken;
            if (prev is null)
            {
                // First sample — logits are already in the buffer from the
                // caller's prior decode. Take the lock anyway so we're not
                // racing with another session's decode between now and our
                // sampler.Sample call.
                nextToken = await _context.WithDecodeLockAsync(
                    () => _sampler.Sample(_context), cancellationToken).ConfigureAwait(false);
            }
            else
            {
                int prevTok = prev.Value;
                nextToken = await _context.WithDecodeLockAsync(() =>
                {
                    DecodeSingleToken(prevTok);
                    return _sampler.Sample(_context);
                }, cancellationToken).ConfigureAwait(false);
            }

            if (vocab.IsEndOfGeneration(nextToken))
            {
                int finalChars = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
                if (finalChars > 0) yield return new string(charBuf, 0, finalChars);
                LastStopReason = LlamaStopReason.EndOfGeneration;
                yield break;
            }

            var pieceBytes = GetPieceBytes(nextToken, renderSpecialPieces);
            if (pieceBytes.Length > 0)
            {
                int charCount = decoder.GetChars(pieceBytes, charBuf, flush: false);
                while (charCount == 0 && pieceBytes.Length > charBuf.Length)
                {
                    Array.Resize(ref charBuf, pieceBytes.Length * 2);
                    charCount = decoder.GetChars(pieceBytes, charBuf, flush: false);
                }
                if (charCount > 0) yield return new string(charBuf, 0, charCount);
            }

            if (_sampler.HasGrammar && _sampler.IsGrammarSatisfied(vocab))
            {
                int finalChars = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
                if (finalChars > 0) yield return new string(charBuf, 0, finalChars);
                LastStopReason = LlamaStopReason.GrammarSatisfied;
                yield break;
            }

            prev = nextToken;
            emitted++;
        }

        // Hit maxTokens: the last emitted token still needs to land in KV so
        // the context's state matches the contract the pre-multi-session
        // path promised ("every emitted token ends up decoded unless we
        // stopped on EOG"). Decode it under the lock.
        if (prev.HasValue)
        {
            int finalPrev = prev.Value;
            await _context.WithDecodeLockAsync(
                () => DecodeSingleToken(finalPrev), cancellationToken).ConfigureAwait(false);
        }

        int trailing = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
        if (trailing > 0) yield return new string(charBuf, 0, trailing);
        LastStopReason = LlamaStopReason.MaxTokens;
    }

    /// <summary>
    /// Like <see cref="GenerateAsync(string, int, bool, bool, bool, CancellationToken)"/>
    /// but starts from a pre-tokenized prompt. Useful when the host assembles
    /// tokens directly (e.g. from a chat-template pass) or splices prior
    /// conversation tokens.
    /// </summary>
    /// <param name="firstNewIndex">
    /// Index into <paramref name="promptTokens"/> of the first token not yet
    /// decoded into the KV cache. Tokens at <c>[0, firstNewIndex)</c> are
    /// assumed to be already present and are skipped by the prompt batch —
    /// the caller is responsible for having decoded them previously (and for
    /// having trimmed any stale suffix via
    /// <see cref="LlamaContext.RemoveSequenceRange"/> if the cache had more
    /// tokens than the common prefix). The sampler is still primed with the
    /// full <paramref name="promptTokens"/>, so penalty/DRY state reflects the
    /// whole visible prompt. Default 0 — decode the whole prompt fresh.
    /// </param>
    public async IAsyncEnumerable<string> GenerateAsync(
        IReadOnlyList<int> promptTokens,
        int maxTokens = 512,
        bool renderSpecialPieces = false,
        int firstNewIndex = 0,
        Action<int>? onTokenDecoded = null,
        int logprobsTopN = 0,
        Action<TokenLogprobInfo>? onLogprobs = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);
        if (promptTokens.Count == 0)
        {
            throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokens));
        }
        if (firstNewIndex < 0 || firstNewIndex > promptTokens.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(firstNewIndex),
                $"firstNewIndex must be in [0, {promptTokens.Count}]; got {firstNewIndex}.");
        }

        var vocab = _context.Model.Vocab;

        // 1) Ingest the suffix of the prompt that isn't already in the KV cache.
        //    llama_batch_get_one references the caller's array directly, so we
        //    copy to a dedicated array we control and pin for the call. When
        //    the full prompt is already cached (firstNewIndex == Count),
        //    llama_decode would reject an empty batch — the "must evaluate at
        //    least one token" constraint — so we back off one token in that
        //    case, mirroring llama.cpp/tools/server's handling.
        int skipCount = firstNewIndex;
        if (skipCount == promptTokens.Count && skipCount > 0) skipCount -= 1;

        var suffixArray = new int[promptTokens.Count - skipCount];
        for (int i = 0; i < suffixArray.Length; i++)
        {
            suffixArray[i] = promptTokens[skipCount + i];
        }

        // 1-2) Prompt decode and first sample under a single lock acquire. A
        // second session decoding between our DecodePromptBatch and our
        // first sampler.Sample would give us their logits instead of ours —
        // serializing the pair eliminates that window.
        cancellationToken.ThrowIfCancellationRequested();
        int firstToken = await _context.WithDecodeLockAsync(() =>
        {
            DecodePromptBatch(suffixArray);
            // Prime the sampler chain with the full prompt — even the part
            // that was cached. Mirrors llama.cpp's common_sampler_accept with
            // accept_grammar=false: penalties / DRY must see the prompt so
            // they treat repetition of prompt words as repetition. Grammar
            // is intentionally NOT primed here — it only constrains
            // generation, and since e1423ef the grammar is held outside the
            // chain, so this loop can't accidentally touch it.
            foreach (var t in promptTokens) _sampler.Accept(t);
            // llama_sampler_sample already does apply + select + accept
            // internally. Do NOT call _sampler.Accept(nextToken) after —
            // that double-advances stateful stages (grammar, penalties).
            int chosen = _sampler.Sample(_context);
            // logprobs (if requested): logits at -1 are still valid here
            // since the sampler doesn't advance state, so capture before
            // we leave the lock.
            if (onLogprobs is not null)
            {
                onLogprobs(ComputeLogprobsForChosen(chosen, logprobsTopN));
            }
            return chosen;
        }, cancellationToken).ConfigureAwait(false);

        // 3) Sampling loop. Each iteration: emit the most recent sample;
        //    if we're continuing, atomically decode that sample and produce
        //    the next one under the same lock acquire.
        var decoder = Encoding.UTF8.GetDecoder();
        var charBuf = new char[64];
        int emitted = 0;
        LastStopReason = LlamaStopReason.None;
        int currentToken = firstToken;

        while (emitted < maxTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (vocab.IsEndOfGeneration(currentToken))
            {
                // EOG token is NOT decoded into KV — matches the pre-multi-
                // session contract that the model's requested stop point
                // doesn't land as a live token in the cache.
                int finalChars = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
                if (finalChars > 0) yield return new string(charBuf, 0, finalChars);
                LastStopReason = LlamaStopReason.EndOfGeneration;
                yield break;
            }

            // TokenToPiece handles UTF-8 correctly but a single token may be
            // a partial multi-byte sequence (very common for CJK, emoji).
            // Feed the raw bytes into an incremental UTF-8 decoder so we
            // only yield complete characters — partial bytes stay buffered
            // until the next token completes them.
            var pieceBytes = GetPieceBytes(currentToken, renderSpecialPieces);
            if (pieceBytes.Length > 0)
            {
                int charCount = decoder.GetChars(pieceBytes, charBuf, flush: false);
                while (charCount == 0 && pieceBytes.Length > charBuf.Length)
                {
                    Array.Resize(ref charBuf, pieceBytes.Length * 2);
                    charCount = decoder.GetChars(pieceBytes, charBuf, flush: false);
                }
                if (charCount > 0) yield return new string(charBuf, 0, charCount);
            }

            // Grammar-termination gate: if the sampler has a grammar stage
            // and the grammar now permits only EOG tokens, stop cleanly
            // instead of letting the next Sample/Accept crash the process
            // with the "empty grammar stack" runtime_error.
            if (_sampler.HasGrammar && _sampler.IsGrammarSatisfied(vocab))
            {
                int finalChars = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
                if (finalChars > 0) yield return new string(charBuf, 0, finalChars);
                LastStopReason = LlamaStopReason.GrammarSatisfied;
                yield break;
            }

            emitted++;
            if (emitted >= maxTokens) break;

            int tokenToDecode = currentToken;
            currentToken = await _context.WithDecodeLockAsync(() =>
            {
                DecodeSingleToken(tokenToDecode);
                int chosen = _sampler.Sample(_context);
                if (onLogprobs is not null)
                {
                    onLogprobs(ComputeLogprobsForChosen(chosen, logprobsTopN));
                }
                return chosen;
            }, cancellationToken).ConfigureAwait(false);
            // onTokenDecoded fires outside the lock so a slow callback can't
            // throttle other sessions. By the time we're here the token is
            // already committed to KV.
            onTokenDecoded?.Invoke(tokenToDecode);
        }

        // Hit maxTokens: the last emitted token is in `currentToken` and has
        // NOT been decoded yet. The pre-multi-session contract was "every
        // emitted token lands in KV unless we stopped on EOG", so decode it
        // now so the context's state matches what callers expect.
        await _context.WithDecodeLockAsync(
            () => DecodeSingleToken(currentToken), cancellationToken).ConfigureAwait(false);
        onTokenDecoded?.Invoke(currentToken);

        // Flush any residual buffered bytes if we hit the maxTokens cap mid-char.
        int trailing2 = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
        if (trailing2 > 0) yield return new string(charBuf, 0, trailing2);
        LastStopReason = LlamaStopReason.MaxTokens;
    }

    private unsafe void DecodePromptBatch(int[] tokens)
    {
        // llama.cpp asserts n_tokens_all <= cparams.n_batch inside
        // llama_decode and crashes the process with GGML_ASSERT if we exceed
        // it — an uncatchable abort. Chunk into ≤ n_batch slices and decode
        // each sequentially.
        //
        // For the legacy seq_id=0 path we stay on llama_batch_get_one so the
        // fast default-path behaviour (no managed-side batch allocation) is
        // preserved exactly. For any other seq_id we build an explicit batch
        // via llama_batch_init + manual seq_id / position fill, since
        // batch_get_one hard-codes seq 0.
        int batchCap = Math.Max(1, _context.LogicalBatchSize);
        int offset = 0;

        if (_sequenceId == 0)
        {
            while (offset < tokens.Length)
            {
                int take = Math.Min(batchCap, tokens.Length - offset);
                fixed (int* tokPtr = &tokens[offset])
                {
                    var batch = NativeMethods.llama_batch_get_one(tokPtr, take);
                    var rc = NativeMethods.llama_decode(_context.Handle.DangerousHandle, batch);
                    if (rc != 0)
                    {
                        throw new LlamaException(
                            nameof(NativeMethods.llama_decode), rc,
                            $"llama_decode returned {rc} for a {take}-token prompt chunk " +
                            $"(offset {offset}/{tokens.Length}). " +
                            (rc == 1 ? "Code 1 = no KV slot; reduce prompt size or increase context." : ""));
                    }
                }
                offset += take;
            }
            return;
        }

        // Non-zero sequence: managed-side batch. Position starts at the
        // sequence's current KV tail (or 0 if empty) and advances per chunk.
        var range = _context.SequencePositionRange(_sequenceId);
        int basePos = range.Maximum.HasValue ? range.Maximum.Value + 1 : 0;

        var batchHandle = NativeMethods.llama_batch_init(batchCap, embd: 0, n_seq_max: 1);
        try
        {
            while (offset < tokens.Length)
            {
                int take = Math.Min(batchCap, tokens.Length - offset);
                PopulateBatch(ref batchHandle, tokens, offset, take, basePos + offset, _sequenceId, logitsOnLast: true);
                var rc = NativeMethods.llama_decode(_context.Handle.DangerousHandle, batchHandle);
                if (rc != 0)
                {
                    throw new LlamaException(
                        nameof(NativeMethods.llama_decode), rc,
                        $"llama_decode returned {rc} for a {take}-token prompt chunk on seq_id {_sequenceId} " +
                        $"(offset {offset}/{tokens.Length}). " +
                        (rc == 1 ? "Code 1 = no KV slot; reduce prompt size or increase context." : ""));
                }
                offset += take;
            }
        }
        finally
        {
            NativeMethods.llama_batch_free(batchHandle);
        }
    }

    private unsafe void DecodeSingleToken(int token)
    {
        if (_sequenceId == 0)
        {
            var tokens = stackalloc int[1]; tokens[0] = token;
            var batch = NativeMethods.llama_batch_get_one(tokens, 1);
            var rc = NativeMethods.llama_decode(_context.Handle.DangerousHandle, batch);
            if (rc != 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_decode), rc,
                    $"llama_decode returned {rc} during generation step.");
            }
            return;
        }

        var range = _context.SequencePositionRange(_sequenceId);
        int nextPos = range.Maximum.HasValue ? range.Maximum.Value + 1 : 0;

        var batchHandle = NativeMethods.llama_batch_init(1, embd: 0, n_seq_max: 1);
        try
        {
            var arr = new[] { token };
            PopulateBatch(ref batchHandle, arr, 0, 1, nextPos, _sequenceId, logitsOnLast: true);
            var rc2 = NativeMethods.llama_decode(_context.Handle.DangerousHandle, batchHandle);
            if (rc2 != 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_decode), rc2,
                    $"llama_decode returned {rc2} during generation step on seq_id {_sequenceId}.");
            }
        }
        finally
        {
            NativeMethods.llama_batch_free(batchHandle);
        }
    }

    private static unsafe void PopulateBatch(
        ref Native.llama_batch batch, int[] tokens, int offset, int count,
        int basePos, int seqId, bool logitsOnLast)
    {
        // batch was allocated via llama_batch_init with n_seq_max = 1, so
        // each seq_id[i] points to a one-element array we can dereference
        // directly. Pass by ref so the caller sees the updated n_tokens —
        // llama_batch is a value struct; without ref we'd mutate a copy and
        // llama_decode would see stale n_tokens.
        batch.n_tokens = count;
        var tokPtr   = (int*)batch.token;
        var posPtr   = (int*)batch.pos;
        var nSeqPtr  = (int*)batch.n_seq_id;
        var seqIdArr = (int**)batch.seq_id;
        var logits   = (sbyte*)batch.logits;

        for (int i = 0; i < count; i++)
        {
            tokPtr[i]     = tokens[offset + i];
            posPtr[i]     = basePos + i;
            nSeqPtr[i]    = 1;
            seqIdArr[i][0] = seqId;
            logits[i]     = (sbyte)(logitsOnLast && i == count - 1 ? 1 : 0);
        }
    }

    private byte[] GetPieceBytes(int token, bool renderSpecial)
    {
        // Equivalent of LlamaVocab.TokenToPiece but returns raw bytes so we
        // can feed them to the incremental UTF-8 decoder. Copying into a
        // managed byte[] is cheap — single-token pieces are ≤16 bytes in
        // practice.
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
                _context.Model.Vocab.Handle, token, bufPtr, buf.Length, lstrip: 0, special: renderSpecial);
        }
    }

    /// <summary>
    /// Compute the natural log-softmax over the model's last-position
    /// logits, look up <paramref name="chosenToken"/>'s log-probability,
    /// and return the top <paramref name="topN"/> alternatives by raw
    /// logit. Caller must invoke this under the context's decode lock —
    /// <c>llama_get_logits_ith</c> returns a pointer into the context's
    /// internal buffer that's valid only until the next decode.
    /// </summary>
    private unsafe TokenLogprobInfo ComputeLogprobsForChosen(int chosenToken, int topN)
    {
        int nVocab = _context.Model.Vocab.TokenCount;
        float* logits = Native.NativeMethods.llama_get_logits_ith(
            _context.Handle.DangerousHandle, -1);
        if (logits is null)
        {
            // No logits at -1 — shouldn't happen on the post-decode path
            // but guard rather than dereference null.
            return new TokenLogprobInfo(chosenToken, 0f, Array.Empty<TokenLogprobAlternative>());
        }

        // log-sum-exp trick for numerical stability.
        float maxLogit = logits[0];
        for (int i = 1; i < nVocab; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }
        double sum = 0;
        for (int i = 0; i < nVocab; i++) sum += Math.Exp(logits[i] - maxLogit);
        float logZ = (float)(maxLogit + Math.Log(sum));

        float chosenLogprob = chosenToken >= 0 && chosenToken < nVocab
            ? logits[chosenToken] - logZ
            : 0f;

        if (topN <= 0)
        {
            return new TokenLogprobInfo(chosenToken, chosenLogprob, Array.Empty<TokenLogprobAlternative>());
        }

        // Top-N by raw logit, using a min-heap of size N. Equivalent
        // ordering to top-N by log-probability (logZ is constant).
        // PriorityQueue<TElement, TPriority>.Peek() returns the
        // ELEMENT, not the priority — we compare against the priority
        // via TryPeek out-parameter to avoid the silent bug.
        int n = Math.Min(topN, nVocab);
        var heap = new PriorityQueue<int, float>(n);
        for (int i = 0; i < nVocab; i++)
        {
            if (heap.Count < n)
            {
                heap.Enqueue(i, logits[i]);
            }
            else
            {
                heap.TryPeek(out _, out var smallest);
                if (logits[i] > smallest)
                {
                    heap.Dequeue();
                    heap.Enqueue(i, logits[i]);
                }
            }
        }
        var top = new TokenLogprobAlternative[heap.Count];
        for (int i = top.Length - 1; i >= 0; i--)
        {
            int id = heap.Dequeue();
            top[i] = new TokenLogprobAlternative(id, logits[id] - logZ);
        }
        return new TokenLogprobInfo(chosenToken, chosenLogprob, top);
    }
}

/// <summary>
/// Per-token log-probability info reported via the
/// <c>onLogprobs</c> callback on
/// <see cref="LlamaGenerator.GenerateAsync(IReadOnlyList{int},int,bool,int,Action{int}?,int,Action{TokenLogprobInfo}?,CancellationToken)"/>.
/// </summary>
/// <param name="TokenId">The token id that was sampled at this step.</param>
/// <param name="Logprob">Natural log of the post-softmax probability for the sampled token under the full vocabulary.</param>
/// <param name="TopAlternatives">Top-N candidates by logit, each with its own logprob. Empty when the caller didn't request top-N.</param>
public readonly record struct TokenLogprobInfo(
    int TokenId,
    float Logprob,
    IReadOnlyList<TokenLogprobAlternative> TopAlternatives);

public readonly record struct TokenLogprobAlternative(int TokenId, float Logprob);

/// <summary>
/// Why generation stopped. Distinct from "how the caller observed the stop"
/// (cancellation, exception): this reflects the sampler / generator's own
/// termination path. Callers use it to decide whether extending the reply
/// via a continue-loop is meaningful — e.g. <see cref="EndOfGeneration"/>
/// means the model emitted a natural EOG token, so extending would just
/// push past its intended stop point.
/// </summary>
public enum LlamaStopReason
{
    /// <summary>Generation hasn't been run, or was interrupted before recording a reason.</summary>
    None = 0,
    /// <summary>Model emitted a tokenizer-defined end-of-generation token (<c>&lt;|eot_id|&gt;</c>, <c>&lt;|im_end|&gt;</c>, etc.).</summary>
    EndOfGeneration = 1,
    /// <summary>Hit the <c>maxTokens</c> budget without ever sampling an EOG.</summary>
    MaxTokens = 2,
    /// <summary>A grammar stage reached a state that only permits EOG; we stopped cleanly rather than sample illegal tokens.</summary>
    GrammarSatisfied = 3,
    /// <summary>Cancellation observed by the caller. The generator itself doesn't set this; it's a hint for downstream consumers.</summary>
    Cancelled = 4,
}
