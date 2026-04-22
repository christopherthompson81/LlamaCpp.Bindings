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
    public static LlamaSampler Build(LlamaModel model, LlamaVocab vocab, SamplerSettings s)
    {
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
            b.WithXtc(xtcProb, s.XtcThreshold, minKeep: 1, seed: s.Seed);
        }

        if (!string.IsNullOrWhiteSpace(s.GbnfGrammar))
        {
            b.WithGrammar(vocab, new LlamaGrammar(s.GbnfGrammar!, s.GrammarStartRule));
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
                b.WithMirostat(vocabSize: (int)vocab.TokenCount, seed: s.Seed, tau: s.MirostatTau, eta: s.MirostatEta);
                break;
            case MirostatMode.V2:
                b.WithMirostatV2(seed: s.Seed, tau: s.MirostatTau, eta: s.MirostatEta);
                break;
            case MirostatMode.Off:
            default:
                b.WithDistribution(s.Seed);
                break;
        }

        return b.Build();
    }
}
