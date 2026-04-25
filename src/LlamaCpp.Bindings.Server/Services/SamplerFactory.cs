namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Per-request sampler knobs. One record shared by <c>/v1/chat/completions</c>
/// and <c>/completion</c> so both endpoints speak the same dialect and
/// <see cref="SamplerFactory.Build"/> has one place to implement chain
/// ordering. All fields are nullable — omitted means "binding default."
/// </summary>
public sealed record SamplerParams
{
    // ----- terminal / temperature -----
    public float? Temperature { get; init; }
    public uint? Seed { get; init; }

    /// <summary>0 = off, 1 = Mirostat v1, 2 = Mirostat v2. Mirostat overrides truncation and temperature.</summary>
    public int? Mirostat { get; init; }
    public float? MirostatTau { get; init; }
    public float? MirostatEta { get; init; }

    // ----- truncation -----
    public int? TopK { get; init; }
    public float? TopP { get; init; }
    public float? MinP { get; init; }
    public float? TypicalP { get; init; }
    public float? TopNSigma { get; init; }

    // ----- XTC (exclude top choices) -----
    public float? XtcProbability { get; init; }
    public float? XtcThreshold { get; init; }

    // ----- DRY (don't repeat yourself) -----
    public float? DryMultiplier { get; init; }
    public float? DryBase { get; init; }
    public int? DryAllowedLength { get; init; }
    public int? DryPenaltyLastN { get; init; }
    public IReadOnlyList<string>? DrySequenceBreakers { get; init; }

    // ----- penalties -----
    public float? RepeatPenalty { get; init; }
    public float? FrequencyPenalty { get; init; }
    public float? PresencePenalty { get; init; }
    public int? RepeatLastN { get; init; }

    // ----- logit bias -----
    public IReadOnlyDictionary<string, float>? LogitBias { get; init; }

    /// <summary>
    /// Pre-resolved grammar (GBNF + root rule name). When set, the sampler
    /// constrains generation to strings accepted by this grammar via
    /// rejection sampling. Resolution of the caller's
    /// <c>response_format</c> / <c>json_schema</c> / <c>grammar</c> fields
    /// is the endpoint's job (see <see cref="GrammarFactory"/>); this
    /// record only carries the output.
    /// </summary>
    public LlamaGrammar? Grammar { get; init; }
}

/// <summary>
/// Builds a <see cref="LlamaSampler"/> from a <see cref="SamplerParams"/>.
/// Encapsulates chain ordering + the handful of edge cases that would
/// otherwise be repeated in every endpoint.
/// </summary>
public static class SamplerFactory
{
    /// <summary>
    /// Compose a sampler chain from the request's knobs. Chain order:
    /// <list type="number">
    ///   <item>logit bias (so later stages see adjusted logits)</item>
    ///   <item>penalties (history-based, read-mutate logits)</item>
    ///   <item>DRY (pattern-based repetition suppression)</item>
    ///   <item>truncation: top-k → top-p → min-p → typical → top-n-σ → XTC</item>
    ///   <item>temperature</item>
    ///   <item>terminal: greedy (temp ≤ 0) / distribution / mirostat</item>
    /// </list>
    /// Mirostat (v1 or v2), when active, supplants truncation and the
    /// temperature stage — it runs its own adaptive-perplexity pick and
    /// wouldn't cooperate with a pre-truncated candidate set.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SamplerParams.LogitBias"/> contains a
    /// non-numeric key or a token id out of vocab range, or when
    /// <see cref="SamplerParams.Mirostat"/> is not 0/1/2. The caller
    /// should surface this as HTTP 400.
    /// </exception>
    public static LlamaSampler Build(LlamaModel model, SamplerParams p)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(p);

        var vocab = model.Vocab;
        var b = new LlamaSamplerBuilder();

        // 0. Grammar — held outside the chain by LlamaSamplerBuilder; does
        //    not participate in chain ordering. Attach first so failures
        //    (invalid GBNF) surface before we've wasted per-token cost on
        //    building the rest of the chain.
        if (p.Grammar is LlamaGrammar grammar)
        {
            b = b.WithGrammar(vocab, grammar);
        }

        // 1. Logit bias — applied first so downstream sees the biased
        //    values. Validate keys/ids up front so the request can fail
        //    fast before we hold a pool slot.
        if (p.LogitBias is { Count: > 0 })
        {
            var parsed = ParseLogitBias(p.LogitBias, vocab);
            b = b.WithLogitBias(vocab, parsed);
        }

        // 2. Penalties — only include the stage when at least one knob
        //    diverges from the "off" default. The chain pays a small
        //    per-token cost per stage even when the stage is a no-op.
        bool hasPenalties =
            (p.RepeatPenalty is float r && Math.Abs(r - 1.0f) > float.Epsilon) ||
            (p.FrequencyPenalty is float fp && fp != 0.0f) ||
            (p.PresencePenalty is float pp && pp != 0.0f);
        if (hasPenalties)
        {
            b = b.WithPenalties(
                lastN:     p.RepeatLastN ?? 64,
                repeat:    p.RepeatPenalty ?? 1.0f,
                frequency: p.FrequencyPenalty ?? 0.0f,
                presence:  p.PresencePenalty ?? 0.0f);
        }

        // 3. DRY — off when multiplier is null or ≤ 0 (llama.cpp's
        //    convention). Defaults match common_sampler_params.
        if (p.DryMultiplier is float dm && dm > 0f)
        {
            b = b.WithDry(
                vocab:           vocab,
                contextTrainSize: model.TrainingContextSize,
                multiplier:      dm,
                dryBase:         p.DryBase ?? 1.75f,
                allowedLength:   p.DryAllowedLength ?? 2,
                penaltyLastN:    p.DryPenaltyLastN ?? -1,
                sequenceBreakers: p.DrySequenceBreakers);
        }

        // Mirostat bypasses truncation + temperature; pick the terminal
        // now if the caller asked for it.
        int mirostat = p.Mirostat ?? 0;
        if (mirostat is 1 or 2)
        {
            return BuildMirostatTerminal(b, vocab, mirostat, p);
        }
        if (mirostat != 0)
        {
            throw new ArgumentException(
                $"mirostat must be 0, 1, or 2; got {mirostat}.", nameof(p));
        }

        // 4. Truncation stages. Order follows llama-server's default
        //    sampler_seq (top-k → top-p → min-p → typical → top-n-σ → XTC).
        if (p.TopK is int k && k > 0) b = b.WithTopK(k);
        if (p.TopP is float tp && tp is > 0f and < 1f) b = b.WithTopP(tp);
        if (p.MinP is float mp && mp > 0f) b = b.WithMinP(mp);
        if (p.TypicalP is float ty && ty is > 0f and < 1f) b = b.WithTypical(ty);
        if (p.TopNSigma is float ns && ns > 0f) b = b.WithTopNSigma(ns);
        if (p.XtcProbability is float xp && xp > 0f)
        {
            b = b.WithXtc(
                probability: xp,
                threshold:   p.XtcThreshold ?? 0.1f,
                minKeep:     1,
                seed:        p.Seed ?? 0u);
        }

        // 5. Temperature → 6. Terminal.
        float temp = p.Temperature ?? 0f;
        if (temp <= 0f)
        {
            // Temperature 0 collapses to greedy. Short-circuit so we don't
            // stack a degenerate temp stage before a distribution sampler.
            return b.WithGreedy().Build();
        }
        return b.WithTemperature(temp).WithDistribution(p.Seed ?? 0u).Build();
    }

    private static LlamaSampler BuildMirostatTerminal(
        LlamaSamplerBuilder b, LlamaVocab vocab, int mirostat, SamplerParams p)
    {
        uint seed = p.Seed ?? 0u;
        float tau = p.MirostatTau ?? 5.0f;
        float eta = p.MirostatEta ?? 0.1f;
        if (mirostat == 1)
        {
            return b.WithMirostat(vocab.TokenCount, seed, tau, eta).Build();
        }
        return b.WithMirostatV2(seed, tau, eta).Build();
    }

    private static (int Token, float Bias)[] ParseLogitBias(
        IReadOnlyDictionary<string, float> biases, LlamaVocab vocab)
    {
        var result = new (int, float)[biases.Count];
        int i = 0;
        foreach (var (key, bias) in biases)
        {
            if (!int.TryParse(key, System.Globalization.NumberStyles.Integer,
                              System.Globalization.CultureInfo.InvariantCulture, out int tokenId))
            {
                throw new ArgumentException(
                    $"logit_bias key '{key}' is not a valid token id (integer).");
            }
            if (tokenId < 0 || tokenId >= vocab.TokenCount)
            {
                throw new ArgumentException(
                    $"logit_bias token id {tokenId} is out of range [0, {vocab.TokenCount}).");
            }
            result[i++] = (tokenId, bias);
        }
        return result;
    }
}
