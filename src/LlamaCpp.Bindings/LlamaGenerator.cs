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

    public LlamaGenerator(LlamaContext context, LlamaSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(sampler);
        _context = context;
        _sampler = sampler;
    }

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

        await foreach (var piece in GenerateAsync(promptTokens, maxTokens, renderSpecialPieces, cancellationToken)
                                       .ConfigureAwait(false))
        {
            yield return piece;
        }
    }

    /// <summary>
    /// Like <see cref="GenerateAsync(string, int, bool, bool, bool, CancellationToken)"/>
    /// but starts from a pre-tokenized prompt. Useful when the host assembles
    /// tokens directly (e.g. from a chat-template pass) or splices prior
    /// conversation tokens.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        IReadOnlyList<int> promptTokens,
        int maxTokens = 512,
        bool renderSpecialPieces = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);
        if (promptTokens.Count == 0)
        {
            throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokens));
        }

        var vocab = _context.Model.Vocab;

        // 1) Ingest the prompt as a single batch. llama_batch_get_one references
        //    the caller's array directly, so we copy to a dedicated array we
        //    control and pin for the call. Running the blocking native work on
        //    a pool thread keeps UI callers responsive.
        var promptArray = promptTokens as int[] ?? promptTokens.ToArray();

        cancellationToken.ThrowIfCancellationRequested();
        await Task.Run(() => DecodePromptBatch(promptArray), cancellationToken).ConfigureAwait(false);

        // 2) Sampling loop. At each step: sample one token from the logits of
        //    the last decoded position; accept it into sampler state; exit on
        //    EOG; otherwise decode the single-token batch and emit the piece.
        var decoder = Encoding.UTF8.GetDecoder();
        var charBuf = new char[64];
        var byteBuf = new byte[1];
        int emitted = 0;

        while (emitted < maxTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // llama_sampler_sample already does apply + select + accept
            // internally. Do NOT call _sampler.Accept(nextToken) after — that
            // double-advances stateful stages (grammar, penalties). The
            // double-accept on grammar was fatal ("empty grammar stack"
            // runtime_error) once the grammar got close to completion.
            int nextToken = await Task.Run(() => _sampler.Sample(_context), cancellationToken).ConfigureAwait(false);

            if (vocab.IsEndOfGeneration(nextToken))
            {
                // Flush any bytes buffered in the decoder (rare at EOG but possible).
                int finalChars = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
                if (finalChars > 0) yield return new string(charBuf, 0, finalChars);
                yield break;
            }

            // TokenToPiece handles UTF-8 correctly but a single token may be a
            // partial multi-byte sequence (very common for CJK, emoji). Feed
            // the raw bytes into an incremental UTF-8 decoder so we only yield
            // complete characters — partial bytes stay buffered until the
            // next token completes them.
            var pieceBytes = GetPieceBytes(nextToken, renderSpecialPieces);
            if (pieceBytes.Length > 0)
            {
                int charCount = decoder.GetChars(pieceBytes, charBuf, flush: false);
                // Buffer may need to grow for unusually long pieces (thousand-character
                // tokens don't exist in practice, but be defensive).
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
                yield break;
            }

            // Feed the accepted token back into the context as a 1-token batch
            // so the next decode attends over it.
            await Task.Run(() => DecodeSingleToken(nextToken), cancellationToken).ConfigureAwait(false);

            emitted++;
        }

        // Flush any residual buffered bytes if we hit the maxTokens cap mid-char.
        int trailing = decoder.GetChars(ReadOnlySpan<byte>.Empty, charBuf.AsSpan(), flush: true);
        if (trailing > 0) yield return new string(charBuf, 0, trailing);
        _ = byteBuf; // silence unused warning if the compiler notices
    }

    private unsafe void DecodePromptBatch(int[] tokens)
    {
        fixed (int* tokPtr = tokens)
        {
            var batch = NativeMethods.llama_batch_get_one(tokPtr, tokens.Length);
            var rc = NativeMethods.llama_decode(_context.Handle.DangerousHandle, batch);
            if (rc != 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_decode), rc,
                    $"llama_decode returned {rc} for a {tokens.Length}-token prompt batch. " +
                    (rc == 1 ? "Code 1 = no KV slot; reduce prompt size or increase context." : ""));
            }
        }
    }

    private unsafe void DecodeSingleToken(int token)
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
}
