using LlamaCpp.Bindings.Server.Models;
using LlamaCpp.Bindings.Server.Services;
using Microsoft.AspNetCore.Http;

namespace LlamaCpp.Bindings.Server.Endpoints;

/// <summary>
/// <c>POST /v1/embeddings</c> — OpenAI-compatible embeddings endpoint.
/// When the server is not configured with an embedding model
/// (<see cref="Configuration.ServerOptions.EmbeddingModelPath"/> unset),
/// returns HTTP 501 rather than a silent 404 so clients know the feature
/// exists but is disabled for this deployment.
/// </summary>
public static class EmbeddingsEndpoint
{
    public static async Task Handle(
        HttpContext http,
        EmbeddingsRequest req,
        EmbeddingHost embeddings,
        CancellationToken cancellationToken)
    {
        if (!embeddings.IsAvailable)
        {
            http.Response.StatusCode = StatusCodes.Status501NotImplemented;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message = "This server is not configured with an embedding model. " +
                              "Set LlamaServer:EmbeddingModelPath to enable /v1/embeddings.",
                    type = "not_implemented",
                    code = "embeddings_not_configured",
                },
            }, cancellationToken);
            return;
        }

        if (!string.IsNullOrEmpty(req.EncodingFormat) &&
            !string.Equals(req.EncodingFormat, "float", StringComparison.OrdinalIgnoreCase))
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message = $"encoding_format='{req.EncodingFormat}' is not supported; V1 only supports 'float'.",
                    type = "invalid_request_error",
                    code = "unsupported_encoding_format",
                },
            }, cancellationToken);
            return;
        }

        string[] inputs;
        try
        {
            inputs = req.NormalizeInput();
        }
        catch (ArgumentException ex)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new { message = ex.Message, type = "invalid_request_error" },
            }, cancellationToken);
            return;
        }

        var response = new EmbeddingsResponse
        {
            Model = embeddings.ModelId ?? "",
        };

        // embd_normalize: 2 (default) = L2, 1 = L1, 0 = none.
        // -1 means "let the model decide" — for our binding the model has
        // already L2-normalised by default, so we treat -1 as a no-op
        // (don't re-normalise) the same way upstream does.
        int normalize = req.EmbdNormalize ?? 2;
        bool returnText = req.ReturnText ?? false;

        int totalTokens = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (vec, tokenCount) = await embeddings.EncodeAsync(inputs[i], cancellationToken);
            ApplyNormalization(vec, normalize);
            response.Data.Add(new EmbeddingEntry
            {
                Index = i,
                Embedding = vec,
                Text = returnText ? inputs[i] : null,
            });
            totalTokens += tokenCount;
        }

        response.Usage = new EmbeddingUsage
        {
            PromptTokens = totalTokens,
            TotalTokens = totalTokens,
        };

        http.Response.ContentType = "application/json";
        await http.Response.WriteAsJsonAsync(response, cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Apply the requested normalisation to <paramref name="vec"/> in
    /// place. <c>2</c> = L2 (Euclidean), <c>1</c> = L1 (sum-of-abs),
    /// <c>0</c> or negative = no rescaling. Zero-norm vectors are left
    /// untouched to avoid divide-by-zero.
    /// </summary>
    private static void ApplyNormalization(float[] vec, int mode)
    {
        if (mode <= 0) return;

        double norm = 0;
        if (mode == 1)
        {
            for (int j = 0; j < vec.Length; j++) norm += Math.Abs(vec[j]);
        }
        else
        {
            // Default L2 path — also covers any positive mode we don't
            // recognise, matching upstream's permissive behaviour.
            for (int j = 0; j < vec.Length; j++) norm += vec[j] * vec[j];
            norm = Math.Sqrt(norm);
        }

        if (norm <= 0) return;
        float scale = (float)(1.0 / norm);
        for (int j = 0; j < vec.Length; j++) vec[j] *= scale;
    }
}
