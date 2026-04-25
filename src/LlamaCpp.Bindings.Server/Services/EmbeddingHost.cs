using LlamaCpp.Bindings.Server.Configuration;
using Microsoft.Extensions.Options;

namespace LlamaCpp.Bindings.Server.Services;

/// <summary>
/// Optional singleton that owns a dedicated <see cref="LlamaModel"/> +
/// <see cref="LlamaContext"/> for the <c>/v1/embeddings</c> endpoint.
/// </summary>
/// <remarks>
/// <para>Embedding workloads don't co-habit well with chat inference on
/// one context: the model's pooling type and <see cref="LlamaContextParameters.Embeddings"/>
/// flag are set at context creation and can't be toggled without
/// destabilising generation state. We therefore load a second model when
/// the server wants embeddings support, rather than reusing the chat
/// model's context. Saves arguing with <c>SetEmbeddingsMode</c>
/// round-trips.</para>
///
/// <para>All calls serialize through <see cref="Gate"/> because
/// <see cref="LlamaContext.EncodeForEmbedding"/> clears the KV cache and
/// decodes inside a single call — overlapping requests would clobber
/// each other's vectors. The semaphore is fast; embedding models are
/// small (≤ 1B params, usually ≤ 200M) and single-query throughput on
/// GPU is already hundreds per second, so serial is good enough for V1.
/// </para>
///
/// <para>When <see cref="ServerOptions.EmbeddingModelPath"/> is unset,
/// <see cref="IsAvailable"/> is <c>false</c> and the endpoint returns
/// HTTP 501 without consulting this host.</para>
/// </remarks>
public sealed class EmbeddingHost : IDisposable
{
    private readonly ILogger<EmbeddingHost> _log;
    private readonly LlamaModel? _model;
    private readonly LlamaContext? _context;
    private readonly SemaphoreSlim? _gate;
    private readonly string? _modelId;

    public bool IsAvailable => _model is not null && _context is not null;

    /// <summary>Resolved model id surfaced in <c>/v1/embeddings</c> responses. <c>null</c> when not configured.</summary>
    public string? ModelId => _modelId;

    /// <summary>Embedding dimension (model-specific).</summary>
    public int EmbeddingSize => _model?.EmbeddingSize ?? 0;

    public EmbeddingHost(IOptions<ServerOptions> options, ILogger<EmbeddingHost> log)
    {
        _log = log;
        var opts = options.Value;

        if (string.IsNullOrWhiteSpace(opts.EmbeddingModelPath))
        {
            _log.LogInformation(
                "Embeddings disabled: LlamaServer:EmbeddingModelPath not set. " +
                "/v1/embeddings will return 501.");
            return;
        }
        if (!File.Exists(opts.EmbeddingModelPath))
        {
            throw new FileNotFoundException(
                $"LlamaServer:EmbeddingModelPath='{opts.EmbeddingModelPath}' does not exist.",
                opts.EmbeddingModelPath);
        }

        LlamaBackend.Initialize();

        _log.LogInformation("Loading embedding model from {Path}", opts.EmbeddingModelPath);
        _model = new LlamaModel(opts.EmbeddingModelPath, new LlamaModelParameters
        {
            GpuLayerCount = opts.EmbeddingGpuLayerCount,
            UseMmap = opts.UseMmap,
            UseMlock = opts.UseMlock,
        });

        _context = new LlamaContext(_model, new LlamaContextParameters
        {
            ContextSize = (uint)Math.Max(0, opts.EmbeddingContextSize),
            LogicalBatchSize = (uint)Math.Max(1, opts.EmbeddingBatchSize),
            PhysicalBatchSize = (uint)Math.Max(1, opts.EmbeddingBatchSize),
            MaxSequenceCount = 1,
            Embeddings = true, // context loaded in embeddings mode from the start
        });

        _gate = new SemaphoreSlim(1, 1);
        _modelId = !string.IsNullOrWhiteSpace(opts.EmbeddingModelAlias)
            ? opts.EmbeddingModelAlias!
            : Path.GetFileNameWithoutExtension(opts.EmbeddingModelPath);

        _log.LogInformation(
            "Embedding model loaded: id={Id}, dim={Dim}, pooling={Pooling}",
            _modelId, _model.EmbeddingSize, _context.PoolingType);
    }

    /// <summary>
    /// Tokenize + encode <paramref name="text"/> into a pooled embedding
    /// vector. Serialised through the host's semaphore; concurrent callers
    /// queue. Throws <see cref="InvalidOperationException"/> when the host
    /// is not configured — check <see cref="IsAvailable"/> first.
    /// </summary>
    /// <returns>
    /// A tuple of (embedding vector, prompt-token count). The token count
    /// comes from a separate tokenize call so the response's <c>usage</c>
    /// field has something useful; the vector is produced by
    /// <see cref="LlamaContext.EncodeForEmbedding"/>.
    /// </returns>
    public async Task<(float[] Vector, int TokenCount)> EncodeAsync(
        string text, CancellationToken cancellationToken)
    {
        if (!IsAvailable) throw new InvalidOperationException("Embedding model is not loaded.");

        await _gate!.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await Task.Run(() =>
            {
                // llama-server reports usage.prompt_tokens based on the
                // actually-tokenized input. Tokenize once for counting; the
                // EncodeForEmbedding call below re-tokenizes internally, but
                // that cost is tiny next to the forward pass.
                var tokens = _model!.Vocab.Tokenize(text, addSpecial: true, parseSpecial: false);
                var vec = _context!.EncodeForEmbedding(text, addSpecial: true, parseSpecial: false);
                return (vec, tokens.Length);
            }, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    public void Dispose()
    {
        _gate?.Dispose();
        _context?.Dispose();
        _model?.Dispose();
    }
}
