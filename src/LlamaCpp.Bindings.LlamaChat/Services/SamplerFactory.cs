using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.Services;

internal static class SamplerFactory
{
    /// <summary>
    /// Translate a <see cref="SamplerSettings"/> into a freshly built <see cref="LlamaSampler"/>.
    /// Stage order matches llama.cpp's default chain:
    ///   penalties → DRY → top-n-sigma → top-k → typical → top-p → min-p → XTC → temperature →
    ///   (mirostat | distribution).
    /// The caller owns disposal of the returned sampler.
    /// </summary>
    // Mirror of LLAMA_DEFAULT_SEED — passing this to a seeded llama sampler
    // tells it to draw a random seed at construction time. Since we rebuild
    // the sampler per generation, this yields fresh randomness each turn.
    private const uint LlamaDefaultSeed = 0xFFFFFFFFu;

    public static LlamaSampler Build(LlamaModel model, LlamaVocab vocab, SamplerSettings s)
    {
        var seed = s.Seed ?? LlamaDefaultSeed;
        var b = new LlamaSamplerBuilder();

        var hasPenalty = s.PenaltyRepeat != 1.0f || s.PenaltyFrequency != 0.0f || s.PenaltyPresence != 0.0f;
        if (hasPenalty)
        {
            b.WithPenalties(
                lastN: s.PenaltyLastN,
                repeat: s.PenaltyRepeat,
                frequency: s.PenaltyFrequency,
                presence: s.PenaltyPresence);
        }

        if (s.DryMultiplier > 0f)
        {
            b.WithDry(
                vocab: vocab,
                contextTrainSize: model.TrainingContextSize,
                multiplier: s.DryMultiplier,
                dryBase: s.DryBase,
                allowedLength: s.DryAllowedLength,
                penaltyLastN: s.DryPenaltyLastN);
        }

        if (s.TopNSigma is { } sigma) b.WithTopNSigma(sigma);
        if (s.TopK is { } topK) b.WithTopK(topK);
        if (s.Typical is { } typ) b.WithTypical(typ, minKeep: 1);
        if (s.TopP is { } topP) b.WithTopP(topP, minKeep: 1);
        if (s.MinP is { } minP) b.WithMinP(minP, minKeep: 1);

        if (s.XtcProbability is { } xtcProb && xtcProb > 0f)
        {
            b.WithXtc(xtcProb, s.XtcThreshold, minKeep: 1, seed: seed);
        }

        // Response-format constraint. Dispatch on the mode to figure out
        // what grammar to hand the sampler. Legacy GbnfGrammar (from
        // profiles saved before ResponseFormat existed) wins if the new
        // field is Off and the old one is populated — preserves behaviour
        // on old config files without a migration step.
        var grammar = BuildResponseFormatGrammar(s);
        if (grammar is not null)
        {
            b.WithGrammar(vocab, new LlamaGrammar(grammar, s.GrammarStartRule));
        }

        if (s.DynaTempRange > 0f)
        {
            b.WithExtendedTemperature(s.Temperature, s.DynaTempRange, s.DynaTempExponent);
        }
        else
        {
            b.WithTemperature(s.Temperature);
        }

        switch (s.Mirostat)
        {
            case MirostatMode.V1:
                b.WithMirostat(vocabSize: (int)vocab.TokenCount, seed: seed, tau: s.MirostatTau, eta: s.MirostatEta);
                break;
            case MirostatMode.V2:
                b.WithMirostatV2(seed: seed, tau: s.MirostatTau, eta: s.MirostatEta);
                break;
            case MirostatMode.Off:
            default:
                b.WithDistribution(seed);
                break;
        }

        return b.Build();
    }

    /// <summary>
    /// Pick the grammar source for the sampler based on the response-format
    /// mode, compiling a JSON schema on the fly if needed. Returns null
    /// to skip the grammar stage entirely.
    /// </summary>
    /// <remarks>
    /// Compile errors on a JSON schema throw
    /// <see cref="JsonSchemaConversionException"/>; we let them propagate
    /// so the user sees the specific error, rather than silently dropping
    /// the constraint and generating unconstrained output that looks right
    /// to the eye but fails downstream.
    /// </remarks>
    private static string? BuildResponseFormatGrammar(SamplerSettings s)
    {
        switch (s.ResponseFormat)
        {
            case ResponseFormatMode.Off:
                // Back-compat: old profiles stashed raw GBNF in GbnfGrammar.
                // Honour it so existing files keep working without a migration.
                return string.IsNullOrWhiteSpace(s.GbnfGrammar) ? null : s.GbnfGrammar;

            case ResponseFormatMode.Json:
                // The bindings ship an "any valid JSON" grammar as a static.
                return LlamaGrammar.Json.GbnfSource;

            case ResponseFormatMode.JsonSchema:
                if (string.IsNullOrWhiteSpace(s.ResponseFormatText)) return null;
                return JsonSchemaToGbnf.Convert(s.ResponseFormatText);

            case ResponseFormatMode.Gbnf:
                return string.IsNullOrWhiteSpace(s.ResponseFormatText) ? null : s.ResponseFormatText;

            default:
                return null;
        }
    }
}
