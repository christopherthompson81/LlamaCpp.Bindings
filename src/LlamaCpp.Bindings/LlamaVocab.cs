using System.Text;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// Tokenizer and special-token registry for a loaded model. Lifetime is tied
/// to the owning <see cref="LlamaModel"/> — freeing the model invalidates this
/// instance. Cheap to construct; obtained via <see cref="LlamaModel.Vocab"/>.
/// </summary>
/// <remarks>
/// Special tokens (BOS, EOS, EOT, PAD, SEP, NL) are exposed as nullable ints —
/// models that lack a given token report <see cref="LLAMA_TOKEN_NULL"/> (-1),
/// which we surface as <c>null</c> rather than leak the sentinel into callers.
/// </remarks>
public sealed class LlamaVocab
{
    internal const int LLAMA_TOKEN_NULL = -1;

    private readonly IntPtr _vocab;
    private readonly LlamaModel _owner; // kept alive so vocab doesn't dangle

    /// <summary>Total number of tokens in the vocabulary.</summary>
    public int TokenCount { get; }

    /// <summary>Beginning-of-sequence token, or null if the model has none.</summary>
    public int? Bos { get; }

    /// <summary>End-of-sequence token, or null if the model has none.</summary>
    public int? Eos { get; }

    /// <summary>End-of-turn token (used by chat-tuned models), or null if absent.</summary>
    public int? Eot { get; }

    /// <summary>Newline token, or null if absent.</summary>
    public int? Newline { get; }

    /// <summary>Sentence-separator token, or null if absent.</summary>
    public int? Separator { get; }

    /// <summary>Padding token, or null if absent.</summary>
    public int? Pad { get; }

    internal LlamaVocab(LlamaModel owner, IntPtr vocab)
    {
        _owner = owner;
        _vocab = vocab;

        TokenCount = NativeMethods.llama_vocab_n_tokens(vocab);
        Bos       = NullIfSentinel(NativeMethods.llama_vocab_bos(vocab));
        Eos       = NullIfSentinel(NativeMethods.llama_vocab_eos(vocab));
        Eot       = NullIfSentinel(NativeMethods.llama_vocab_eot(vocab));
        Newline   = NullIfSentinel(NativeMethods.llama_vocab_nl(vocab));
        Separator = NullIfSentinel(NativeMethods.llama_vocab_sep(vocab));
        Pad       = NullIfSentinel(NativeMethods.llama_vocab_pad(vocab));
    }

    private static int? NullIfSentinel(int token) => token == LLAMA_TOKEN_NULL ? null : token;

    /// <summary>
    /// Returns true if <paramref name="token"/> is any end-of-generation marker —
    /// llama.cpp's <c>llama_vocab_is_eog</c> checks for multiple model-specific
    /// terminators, not just EOS. Use this (not an <c>== Eos</c> comparison) to
    /// decide when to stop a generation loop.
    /// </summary>
    public bool IsEndOfGeneration(int token)
    {
        EnsureOwnerAlive();
        return NativeMethods.llama_vocab_is_eog(_vocab, token);
    }

    /// <summary>
    /// Tokenize UTF-8 text into token IDs.
    /// </summary>
    /// <param name="text">Input text. May be empty.</param>
    /// <param name="addSpecial">
    /// If true, BOS (and potentially EOS) tokens are prepended/appended according
    /// to the model's tokenizer config. Typically true for the first chunk of a
    /// prompt; false for continuations.
    /// </param>
    /// <param name="parseSpecial">
    /// If true, special/control token text (e.g. <c>&lt;|im_start|&gt;</c>) is
    /// tokenized as the actual special token rather than as literal text.
    /// Set true when tokenizing a chat-templated prompt.
    /// </param>
    public int[] Tokenize(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        EnsureOwnerAlive();

        if (text.Length == 0)
        {
            // Empty input — still delegate so llama.cpp can decide whether to
            // emit a BOS token (add_special==true + BOS-model).
            return TokenizeCore([], addSpecial, parseSpecial);
        }

        var utf8 = Encoding.UTF8.GetBytes(text);
        return TokenizeCore(utf8, addSpecial, parseSpecial);
    }

    private int[] TokenizeCore(ReadOnlySpan<byte> utf8, bool addSpecial, bool parseSpecial)
    {
        // Heuristic first guess: one token per UTF-8 byte is a generous overestimate
        // for any real tokenizer (they average 3-5 bytes per token). Minimum 16 to
        // avoid a retry on trivially short inputs that pull in a BOS/EOS.
        int cap = Math.Max(16, utf8.Length + 2);
        var buf = new int[cap];
        int n = CallTokenize(utf8, buf, addSpecial, parseSpecial);

        if (n < 0)
        {
            // Buffer too small — |n| is the exact required size.
            cap = -n;
            buf = new int[cap];
            n = CallTokenize(utf8, buf, addSpecial, parseSpecial);
            if (n < 0)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_tokenize),
                    n,
                    $"Tokenization failed after retry: native wanted {-n} tokens but reported overflow again.");
            }
        }

        if (n == buf.Length) return buf;
        var trimmed = new int[n];
        Array.Copy(buf, trimmed, n);
        return trimmed;
    }

    private unsafe int CallTokenize(ReadOnlySpan<byte> utf8, Span<int> tokens, bool addSpecial, bool parseSpecial)
    {
        fixed (byte* textPtr = utf8)
        fixed (int* bufPtr = tokens)
        {
            return NativeMethods.llama_tokenize(
                _vocab,
                textPtr, utf8.Length,
                bufPtr, tokens.Length,
                addSpecial, parseSpecial);
        }
    }

    /// <summary>
    /// Convert a single token id back to its textual piece.
    /// </summary>
    /// <param name="token">Token id; must be in [0, <see cref="TokenCount"/>).</param>
    /// <param name="renderSpecial">
    /// If true, special tokens render as their textual form (e.g. <c>&lt;|eos|&gt;</c>).
    /// If false (the default for user-visible output), they render as empty
    /// strings so the assembled stream matches what a human would read.
    /// </param>
    public string TokenToPiece(int token, bool renderSpecial = false)
    {
        EnsureOwnerAlive();

        // Probe for required size first. llama.cpp returns -needed when the
        // buffer is too small. Start with a stack-friendly buffer and retry.
        const int StackBuf = 256;
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
            return Encoding.UTF8.GetString(heap, 0, n);
        }

        // buf is a stack span — need to materialise before returning.
        return Encoding.UTF8.GetString(buf[..n]);
    }

    private unsafe int CallTokenToPiece(int token, Span<byte> buf, bool renderSpecial)
    {
        fixed (byte* bufPtr = buf)
        {
            return NativeMethods.llama_token_to_piece(
                _vocab, token, bufPtr, buf.Length, lstrip: 0, special: renderSpecial);
        }
    }

    /// <summary>
    /// Best-effort concatenation of <see cref="TokenToPiece"/> over a sequence.
    /// Equivalent to llama.cpp's detokenize; handles multi-byte splits correctly
    /// because each piece is emitted as valid UTF-8 (or a leading continuation
    /// byte that the next call completes).
    /// </summary>
    public string Detokenize(ReadOnlySpan<int> tokens, bool renderSpecial = false)
    {
        if (tokens.IsEmpty) return string.Empty;
        EnsureOwnerAlive();

        var sb = new StringBuilder(tokens.Length * 4);
        foreach (var t in tokens)
        {
            sb.Append(TokenToPiece(t, renderSpecial));
        }
        return sb.ToString();
    }

    internal IntPtr Handle
    {
        get
        {
            EnsureOwnerAlive();
            return _vocab;
        }
    }

    private void EnsureOwnerAlive()
    {
        // The vocab pointer is owned by the model. If the model has been disposed,
        // any native call through our pointer is UB. This check throws a
        // predictable ObjectDisposedException instead.
        _owner.EnsureNotDisposed();
    }
}
