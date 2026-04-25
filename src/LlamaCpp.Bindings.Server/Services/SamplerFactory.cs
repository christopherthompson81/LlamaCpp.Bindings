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

    /// <summary>
    /// Dynamic-temperature stretch (llama.cpp's <c>dynatemp_range</c>).
    /// When &gt; 0, the temperature stage uses
    /// <c>WithExtendedTemperature</c> instead of plain
    /// <c>WithTemperature</c> — the effective temperature flexes around
    /// <see cref="Temperature"/> by ± this amount based on entropy.
    /// </summary>
    public float? DynatempRange { get; init; }

    /// <summary>
    /// Curve shaping for dynamic temperature (llama.cpp's
    /// <c>dynatemp_exponent</c>). 1.0 (default) is linear; higher values
    /// concentrate the stretch on high-entropy positions.
    /// </summary>
    public float? DynatempExponent { get; init; }

    /// <summary>0 = off, 1 = Mirostat v1, 2 = Mirostat v2. Mirostat overrides truncation and temperature.</summary>
    public int? Mirostat { get; init; }
    public float? MirostatTau { get; init; }
    public float? MirostatEta { get; init; }

    /// <summary>
    /// Adaptive-p terminal target. When &gt;= 0, replaces the greedy /
    /// distribution / mirostat terminal with the adaptive-p sampler,
    /// which steers an EMA of selected-token probabilities toward this
    /// target. Mutually exclusive with <see cref="Mirostat"/>; if both
    /// are set, mirostat wins.
    /// </summary>
    public float? AdaptivePTarget { get; init; }

    /// <summary>EMA decay for adaptive-p. 0..0.99; higher = longer memory.</summary>
    public float? AdaptivePDecay { get; init; }

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
    /// Custom ordering for the truncation + temperature stages (llama-server's
    /// <c>samplers</c> field). When null, the factory's documented default
    /// order applies. When non-null, only the named stages run, in the order
    /// given. Stages whose corresponding parameters are absent / disabled are
    /// silently skipped, so a list like <c>["dry","top_k","temperature"]</c>
    /// works even if the request only sets some of those knobs.
    /// </summary>
    /// <remarks>
    /// Penalties and logit-bias remain at the head of the chain regardless;
    /// terminal selection (greedy / distribution / mirostat / adaptive-p)
    /// stays at the tail. Reordering inside the configurable middle is
    /// the entirety of what this knob controls.
    /// </remarks>
    public IReadOnlyList<string>? Samplers { get; init; }

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
    /// Default ordering for the configurable middle of the chain. Matches
    /// llama-server's default sampler_seq with one extension — DRY sits
    /// at the head so pattern-based suppression sees raw probabilities
    /// before any truncation prunes candidates.
    /// </summary>
    public static readonly IReadOnlyList<string> DefaultSamplerOrder = new[]
    {
        "dry",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "top_n_sigma",
        "xtc",
        "temperature",
    };

    /// <summary>
    /// Compose a sampler chain from the request's knobs. Chain order:
    /// <list type="number">
    ///   <item>logit bias (so later stages see adjusted logits)</item>
    ///   <item>penalties (history-based, read-mutate logits)</item>
    ///   <item>configurable middle: dry / top_k / top_p / min_p / typical_p
    ///   / top_n_sigma / xtc / temperature, in the order from
    ///   <see cref="SamplerParams.Samplers"/> or
    ///   <see cref="DefaultSamplerOrder"/></item>
    ///   <item>terminal: mirostat (1/2) / adaptive_p / greedy / distribution</item>
    /// </list>
    /// Mirostat (v1 or v2), when active, supplants the configurable middle
    /// and the temperature stage — it runs its own adaptive-perplexity pick
    /// and wouldn't cooperate with a pre-truncated candidate set. Adaptive-p
    /// is a softer terminal: it preserves the configurable middle but
    /// replaces greedy/distribution at the tail.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SamplerParams.LogitBias"/> contains a
    /// non-numeric key or a token id out of vocab range, when
    /// <see cref="SamplerParams.Mirostat"/> is not 0/1/2, or when
    /// <see cref="SamplerParams.Samplers"/> contains an unknown stage
    /// name. The caller should surface this as HTTP 400.
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

        // Mirostat bypasses the configurable middle — it runs its own
        // adaptive-perplexity pick and wouldn't cooperate with truncated
        // candidates. Pick the terminal now if the caller asked for it.
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

        // 3. Configurable middle. Stages whose params are absent / disabled
        //    silently skip; unknown names throw 400.
        var order = p.Samplers ?? DefaultSamplerOrder;
        foreach (var rawStage in order)
        {
            switch ((rawStage ?? string.Empty).ToLowerInvariant())
            {
                case "dry":
                    if (p.DryMultiplier is float dm && dm > 0f)
                    {
                        b = b.WithDry(
                            vocab:            vocab,
                            contextTrainSize: model.TrainingContextSize,
                            multiplier:       dm,
                            dryBase:          p.DryBase ?? 1.75f,
                            allowedLength:    p.DryAllowedLength ?? 2,
                            penaltyLastN:     p.DryPenaltyLastN ?? -1,
                            sequenceBreakers: p.DrySequenceBreakers);
                    }
                    break;
                case "top_k":
                case "topk":
                    if (p.TopK is int k && k > 0) b = b.WithTopK(k);
                    break;
                case "top_p":
                case "topp":
                    if (p.TopP is float tp && tp is > 0f and < 1f) b = b.WithTopP(tp);
                    break;
                case "min_p":
                case "minp":
                    if (p.MinP is float mp && mp > 0f) b = b.WithMinP(mp);
                    break;
                case "typical_p":
                case "typ_p":
                case "typical":
                    if (p.TypicalP is float ty && ty is > 0f and < 1f) b = b.WithTypical(ty);
                    break;
                case "top_n_sigma":
                case "top_nsigma":
                case "topnsigma":
                    if (p.TopNSigma is float ns && ns > 0f) b = b.WithTopNSigma(ns);
                    break;
                case "xtc":
                    if (p.XtcProbability is float xp && xp > 0f)
                    {
                        b = b.WithXtc(
                            probability: xp,
                            threshold:   p.XtcThreshold ?? 0.1f,
                            minKeep:     1,
                            seed:        p.Seed ?? 0u);
                    }
                    break;
                case "temperature":
                case "temp":
                    AppendTemperature(ref b, p);
                    break;
                default:
                    throw new ArgumentException(
                        $"Unknown sampler stage '{rawStage}'. Allowed: " +
                        "dry, top_k, top_p, min_p, typical_p, top_n_sigma, xtc, temperature.",
                        nameof(p));
            }
        }

        // 4. Terminal. Adaptive-p (when target ≥ 0) supplants greedy /
        //    distribution. Otherwise temp ≤ 0 → greedy, temp > 0 →
        //    distribution. (Temperature itself was added during the
        //    middle pass, if requested.)
        if (p.AdaptivePTarget is float apt && apt >= 0f)
        {
            return b.WithAdaptiveP(
                target: apt,
                decay:  p.AdaptivePDecay ?? 0.95f,
                seed:   p.Seed ?? 0u).Build();
        }
        float temp = p.Temperature ?? 0f;
        if (temp <= 0f)
        {
            return b.WithGreedy().Build();
        }
        return b.WithDistribution(p.Seed ?? 0u).Build();
    }

    /// <summary>
    /// Append the temperature stage. Picks plain
    /// <c>WithTemperature</c> by default and
    /// <c>WithExtendedTemperature</c> when
    /// <see cref="SamplerParams.DynatempRange"/> is positive — the
    /// extended form takes a base temperature plus a stretch range and
    /// exponent and flexes the effective temperature with entropy.
    /// </summary>
    private static void AppendTemperature(ref LlamaSamplerBuilder b, SamplerParams p)
    {
        float temp = p.Temperature ?? 0f;
        if (temp <= 0f) return; // Greedy terminal will swallow this; no temp stage needed.

        if (p.DynatempRange is float range && range > 0f)
        {
            float exponent = p.DynatempExponent ?? 1.0f;
            b = b.WithExtendedTemperature(temp, range, exponent);
        }
        else
        {
            b = b.WithTemperature(temp);
        }
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
