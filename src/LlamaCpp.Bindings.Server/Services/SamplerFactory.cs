namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Shared sampler builder for chat + completion endpoints. Keeps parsing
/// of OpenAI-style sampler knobs (temperature, top-k/p, seed, logit_bias)
/// in one place so the two endpoints can't drift apart on subtle things
/// like "what does temperature 0 mean."
/// </summary>
public static class SamplerFactory
{
    /// <summary>
    /// Build a sampler from the request's common knobs. <paramref name="vocab"/>
    /// is required to validate and apply the logit-bias map; pass the
    /// owning model's vocab.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="logitBias"/> contains a non-numeric key
    /// or a token id outside <c>[0, vocab.TokenCount)</c>. The caller should
    /// translate this into HTTP 400.
    /// </exception>
    public static LlamaSampler Build(
        LlamaVocab vocab,
        float? temperature,
        int? topK,
        float? topP,
        uint? seed,
        IReadOnlyDictionary<string, float>? logitBias)
    {
        var b = new LlamaSamplerBuilder();

        // logit_bias is applied FIRST so temperature / top-k / top-p see the
        // biased logits rather than the raw ones. Mirrors llama.cpp's
        // common_sampler chain ordering.
        if (logitBias is { Count: > 0 })
        {
            var parsed = ParseLogitBias(logitBias, vocab);
            b = b.WithLogitBias(vocab, parsed);
        }

        if (topK is int k && k > 0) b = b.WithTopK(k);
        if (topP is float p && p is > 0f and < 1f) b = b.WithTopP(p);

        // A temperature of 0 (or unset) collapses to greedy — short-circuit
        // so the chain doesn't stack a degenerate temp=0 stage before a
        // distribution sampler that would give the same result slower.
        float temp = temperature ?? 0f;
        if (temp <= 0f)
        {
            return b.WithGreedy().Build();
        }
        return b.WithTemperature(temp).WithDistribution(seed ?? 0u).Build();
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
