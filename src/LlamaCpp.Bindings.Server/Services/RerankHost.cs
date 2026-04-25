using LlamaCpp.Bindings.Server.Configuration;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Optional singleton holding a reranker model + context for the
/// <c>/v1/rerank</c> endpoint. Mirrors <see cref="EmbeddingHost"/>'s
/// shape — a separate model loaded in embeddings mode, with the
/// expectation that the GGUF advertises <see cref="LlamaPoolingType.Rank"/>
/// (or that pooling is otherwise auto-detected to rank). Calls
/// serialize through a semaphore because <see cref="EncodeForEmbedding"/>
/// clears the KV cache internally.
/// </summary>
public sealed class RerankHost : IDisposable
{
    private readonly ILogger<RerankHost> _log;
    private readonly LlamaModel? _model;
    private readonly LlamaContext? _context;
    private readonly SemaphoreSlim? _gate;
    private readonly string? _modelId;

    public bool IsAvailable => _model is not null && _context is not null;

    public string? ModelId => _modelId;

    public RerankHost(IOptions<ServerOptions> options, ILogger<RerankHost> log)
    {
        _log = log;
        var opts = options.Value;

        if (string.IsNullOrWhiteSpace(opts.RerankModelPath))
        {
            _log.LogInformation(
                "Rerank disabled: LlamaServer:RerankModelPath not set. " +
                "/v1/rerank will return 501.");
            return;
        }
        if (!File.Exists(opts.RerankModelPath))
        {
            throw new FileNotFoundException(
                $"LlamaServer:RerankModelPath='{opts.RerankModelPath}' does not exist.",
                opts.RerankModelPath);
        }

        LlamaBackend.Initialize();

        _log.LogInformation("Loading rerank model from {Path}", opts.RerankModelPath);
        _model = new LlamaModel(opts.RerankModelPath, new LlamaModelParameters
        {
            GpuLayerCount = opts.RerankGpuLayerCount,
            UseMmap = opts.UseMmap,
            UseMlock = opts.UseMlock,
        });

        _context = new LlamaContext(_model, new LlamaContextParameters
        {
            ContextSize = (uint)Math.Max(0, opts.RerankContextSize),
            LogicalBatchSize = (uint)Math.Max(1, opts.RerankBatchSize),
            PhysicalBatchSize = (uint)Math.Max(1, opts.RerankBatchSize),
            MaxSequenceCount = 1,
            Embeddings = true,
            // Reranker GGUFs don't always advertise rank pooling in
            // metadata; force it so llama_get_embeddings_seq returns the
            // rank head's score rather than null.
            PoolingType = LlamaPoolingType.Rank,
        });

        _gate = new SemaphoreSlim(1, 1);
        _modelId = !string.IsNullOrWhiteSpace(opts.RerankModelAlias)
            ? opts.RerankModelAlias!
            : Path.GetFileNameWithoutExtension(opts.RerankModelPath);

        _log.LogInformation(
            "Rerank model loaded: id={Id}, pooling={Pooling}, classifierOutputs={Outputs}",
            _modelId, _context.PoolingType, _model.ClassifierOutputCount);

        if (_context.PoolingType != LlamaPoolingType.Rank)
        {
            _log.LogWarning(
                "Rerank model loaded but its pooling type is {Pooling}, not Rank. " +
                "Output scores may not be meaningful — pick a model with a rank head " +
                "(e.g. bge-reranker, jina-reranker).", _context.PoolingType);
        }
    }

    /// <summary>
    /// Score one (query, document) pair. Higher is more relevant. The
    /// raw score's range is model-specific (BGE rerankers commonly emit
    /// values in roughly [-10, 10] and aren't normalised); callers that
    /// want probabilities should sigmoid the result themselves.
    /// </summary>
    /// <returns>The rank head's output for the pair.</returns>
    public async Task<float> ScoreAsync(
        string query, string document, CancellationToken cancellationToken)
    {
        if (!IsAvailable) throw new InvalidOperationException("Rerank model is not loaded.");
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(document);

        // Format as a sentence pair separated by a newline. BGE / Jina
        // rerankers were trained on tokenizer-formatted pairs (typically
        // `[CLS] query [SEP] document [SEP]` with the tokenizer adding
        // the special tokens via addSpecial=true). Plain concatenation
        // with a separator is the convention llama.cpp's own server uses
        // for these models.
        var pair = query + "\n" + document;

        await _gate!.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => ScoreCore(pair), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Manual encoder pass — avoids <see cref="LlamaContext.EncodeForEmbedding"/>
    /// which routes encoder-only BERT-family models (BGE-reranker,
    /// XLMRoberta) through <c>llama_decode</c>. That path emits a
    /// "calling encode() instead" warning and still produces embeddings,
    /// but the status flow is finicky enough that going through
    /// <see cref="LlamaContext.RunEncoder"/> directly is more reliable.
    /// </summary>
    private float ScoreCore(string pair)
    {
        _context!.SetEmbeddingsMode(true);
        _context.ClearKvCache();

        var tokens = _model!.Vocab.Tokenize(pair, addSpecial: true, parseSpecial: false);
        int rc = _context.RunEncoder(tokens);
        if (rc != 0)
        {
            throw new LlamaException(
                "llama_encode", rc,
                $"Reranker encode failed with status {rc}.");
        }
        var vec = _context.GetSequenceEmbedding(0);
        if (vec is null || vec.Length == 0)
        {
            throw new InvalidOperationException(
                "Reranker produced no output. The loaded model may not have a rank pooling head; " +
                $"context reports pooling={_context.PoolingType}, " +
                $"classifierOutputs={_model.ClassifierOutputCount}.");
        }
        return vec[0];
    }

    public void Dispose()
    {
        _gate?.Dispose();
        _context?.Dispose();
        _model?.Dispose();
    }
}
