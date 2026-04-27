namespace LlamaCpp.Bindings;

/// <summary>
/// Generic per-tensor type assignments derived from llama.cpp's
/// hand-tuned <c>llama_tensor_get_type_impl</c> heuristic
/// (<c>llama-quant.cpp:411</c>). Specifically the parts of that
/// heuristic that aren't ftype-specific dressing — the per-layer
/// alternation pattern that protects high-leverage tensors on
/// specific layers (first n/8, last n/8, every third middle layer).
/// </summary>
/// <remarks>
/// <para>
/// Run 15 showed the recipe builder catastrophically under-performs
/// stock when it ignores this pattern: ffn_down on alternating
/// layers must be at Q6_K to retain quality at Q4_K_M-class
/// budgets, and a profile that treats ffn_down as a single category
/// can't express that. By treating stock's per-tensor decisions as
/// a hard floor (the recipe can promote, never demote), we
/// guarantee shipping at-least-as-good-as-stock recipes regardless
/// of profile resolution. The profile then contributes
/// <em>additional</em> per-category optimization on top.
/// </para>
/// <para>
/// This is the Q4_K_M-flavored baseline. We don't try to mirror
/// every ftype's quirks (most are arch-specific or tuned for
/// extreme low-bit targets we don't aim at). What's faithful:
/// <list type="bullet">
///   <item><c>output.weight</c> → Q6_K (matches Q4_K_M's lm_head choice)</item>
///   <item><c>token_embd.weight</c> → Q4_K (matches Q4_K_M)</item>
///   <item>Per-layer <c>attn_v.weight</c> → Q6_K when
///     <see cref="UseMoreBits"/> is true (matches Q4_K_M /
///     Q5_K_M's <c>use_more_bits</c> alternation)</item>
///   <item>Per-layer <c>ffn_down.weight</c> → Q6_K when
///     <see cref="UseMoreBits"/> is true (same)</item>
///   <item>All other tensors: no opinion (recipe controls)</item>
/// </list>
/// </para>
/// </remarks>
public static class LlamaStockBaseline
{
    /// <summary>
    /// Per-layer "give this tensor extra bits" predicate, ported from
    /// <c>llama-quant.cpp:417</c>. True for the first n/8 layers, the
    /// last n/8 layers, and every third layer in the middle.
    /// </summary>
    public static bool UseMoreBits(int iLayer, int nLayers) =>
        iLayer < nLayers / 8
        || iLayer >= 7 * nLayers / 8
        || (iLayer - nLayers / 8) % 3 == 2;

    /// <summary>
    /// Compute the per-tensor stock baseline assignment for the given
    /// model. Returns only the tensors that have an opinion — caller
    /// treats absence as "no baseline opinion, recipe / default
    /// applies".
    /// </summary>
    /// <param name="weightTensors">All weight tensors from the target GGUF (each tensor name + dimensions).</param>
    /// <param name="layerCount">Architecture's block count (typically read from <c>{arch}.block_count</c> metadata).</param>
    public static IReadOnlyDictionary<string, LlamaTensorType> Build(
        IReadOnlyList<(string Name, long[] Dimensions)> weightTensors,
        int layerCount)
    {
        ArgumentNullException.ThrowIfNull(weightTensors);
        if (layerCount <= 0) layerCount = 1;

        var baseline = new Dictionary<string, LlamaTensorType>(StringComparer.Ordinal);

        foreach (var (name, _) in weightTensors)
        {
            if (name == "output.weight")
            {
                baseline[name] = LlamaTensorType.Q6_K;
                continue;
            }
            if (name == "token_embd.weight")
            {
                baseline[name] = LlamaTensorType.Q4_K;
                continue;
            }

            // Per-layer protections require a layer index, which we
            // read from the conventional "blk.<i>." prefix. Tensors
            // without that prefix get no opinion.
            if (!TryParseLayerIndex(name, out var iLayer)) continue;

            // attn_v: protect on use_more_bits layers
            if (name.EndsWith(".attn_v.weight", StringComparison.Ordinal))
            {
                if (UseMoreBits(iLayer, layerCount))
                    baseline[name] = LlamaTensorType.Q6_K;
                continue;
            }

            // ffn_down: protect on use_more_bits layers
            if (name.EndsWith(".ffn_down.weight", StringComparison.Ordinal))
            {
                if (UseMoreBits(iLayer, layerCount))
                    baseline[name] = LlamaTensorType.Q6_K;
                continue;
            }
        }

        return baseline;
    }

    /// <summary>
    /// Parse the layer index from a tensor name like
    /// <c>blk.13.attn_v.weight</c> → <c>13</c>. Returns false for
    /// tensors that don't follow the convention (output, token_embd,
    /// rope freqs, etc.).
    /// </summary>
    private static bool TryParseLayerIndex(string tensorName, out int iLayer)
    {
        iLayer = 0;
        const string prefix = "blk.";
        if (!tensorName.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int dot = tensorName.IndexOf('.', prefix.Length);
        if (dot < 0) return false;
        return int.TryParse(tensorName.AsSpan(prefix.Length, dot - prefix.Length), out iLayer);
    }
}
