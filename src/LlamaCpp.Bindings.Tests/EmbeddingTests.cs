namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Loads a small BGE embedding model on CPU. Separate fixture from
/// <see cref="ModelFixture"/> because embedding models need specific
/// context configuration (embeddings mode, causal=false, appropriate
/// pooling), and from <see cref="GpuGenerationFixture"/> because the
/// model is tiny (~37 MB) and CPU is fine.
/// </summary>
public sealed class EmbeddingModelFixture : IDisposable
{
    private const string DefaultModelPath = "/mnt/data/models/bge-small-en-v1.5-q8_0.gguf";

    public LlamaModel? Model { get; }
    public LlamaContext? Context { get; }

    public EmbeddingModelFixture()
    {
        var path = Environment.GetEnvironmentVariable("LLAMACPP_TEST_EMBEDDING_MODEL");
        if (string.IsNullOrWhiteSpace(path)) path = DefaultModelPath;
        if (!File.Exists(path)) return;

        LlamaBackend.Initialize();
        Model = new LlamaModel(path, new LlamaModelParameters
        {
            GpuLayerCount = 0,   // CPU is plenty for a 33M-param embedding model
            UseMmap = true,
        });
        Context = new LlamaContext(Model, new LlamaContextParameters
        {
            ContextSize = 512,
            LogicalBatchSize = 512,
            PhysicalBatchSize = 512,
            MaxSequenceCount = 1,
            Embeddings = true,
            OffloadKQV = false,
            UseFullSwaCache = false,  // BGE doesn't use SWA; don't force it
        });
    }

    public void SkipMessage()
    {
        Console.WriteLine(
            "SKIP: Embedding tests need a BGE-family GGUF at " +
            "/mnt/data/models/bge-small-en-v1.5-q8_0.gguf (run tools/fetch-test-models.py).");
    }

    public void Dispose()
    {
        Context?.Dispose();
        Model?.Dispose();
    }
}

/// <summary>
/// Tier-2 T2-4 embedding tests. Validates that the embedding extraction
/// pipeline produces sensible vectors: right length, finite, distinct
/// for distinct inputs, higher cosine similarity for semantically close
/// pairs than for unrelated ones.
/// </summary>
public class EmbeddingTests : IClassFixture<EmbeddingModelFixture>
{
    private readonly EmbeddingModelFixture _fx;
    public EmbeddingTests(EmbeddingModelFixture fx) => _fx = fx;

    [Fact]
    public void Model_Reports_Sane_Embedding_Dimensions()
    {
        if (_fx.Model is null) { _fx.SkipMessage(); return; }

        // BGE-small has 384-dim embeddings. Don't hard-code; just verify
        // the value is in the plausible range and positive.
        Assert.InRange(_fx.Model.EmbeddingSize, 64, 4096);
    }

    [Fact]
    public void EncodeForEmbedding_Returns_NEmbd_Length_Vector()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        var vec = _fx.Context.EncodeForEmbedding("Hello, world.");
        Assert.Equal(_fx.Model.EmbeddingSize, vec.Length);

        // All finite.
        foreach (var v in vec)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
        // Not all zeros.
        Assert.Contains(vec, v => v != 0.0f);
    }

    [Fact]
    public void Different_Inputs_Produce_Different_Embeddings()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        var a = _fx.Context.EncodeForEmbedding("The cat sat on the mat.");
        var b = _fx.Context.EncodeForEmbedding("Quantum mechanics describes subatomic particles.");

        Assert.Equal(a.Length, b.Length);
        // Some elements should differ substantially.
        int differingSignificantly = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (MathF.Abs(a[i] - b[i]) > 0.01f) differingSignificantly++;
        }
        Assert.True(differingSignificantly > a.Length / 10,
            $"expected meaningful divergence between very different inputs; only {differingSignificantly}/{a.Length} dims differed");
    }

    [Fact]
    public void Semantic_Similarity_Matches_Intuition()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        // A strong embedding model should rank: (query, relevant) > (query, irrelevant).
        var query    = _fx.Context.EncodeForEmbedding("How do I train a neural network?");
        var relevant = _fx.Context.EncodeForEmbedding("Gradient descent updates model weights to minimise loss.");
        var irrelev  = _fx.Context.EncodeForEmbedding("The capital of France is Paris.");

        double simRelevant   = Cosine(query, relevant);
        double simIrrelevant = Cosine(query, irrelev);

        Assert.True(simRelevant > simIrrelevant,
            $"relevant similarity ({simRelevant:F3}) should exceed irrelevant ({simIrrelevant:F3})");
    }

    [Fact]
    public void Identical_Inputs_Produce_Identical_Embeddings()
    {
        if (_fx.Model is null || _fx.Context is null) { _fx.SkipMessage(); return; }

        var a = _fx.Context.EncodeForEmbedding("test input");
        var b = _fx.Context.EncodeForEmbedding("test input");
        Assert.Equal(a.Length, b.Length);

        // CPU inference is deterministic, so identical inputs → bit-identical
        // outputs. This also transitively validates that ClearKvCache +
        // re-encode produces a clean result.
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i], b[i]);
        }
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot   += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        var denom = Math.Sqrt(normA * normB);
        return denom == 0 ? 0 : dot / denom;
    }
}
