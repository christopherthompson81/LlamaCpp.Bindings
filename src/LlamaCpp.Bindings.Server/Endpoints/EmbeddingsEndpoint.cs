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

        int totalTokens = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (vec, tokenCount) = await embeddings.EncodeAsync(inputs[i], cancellationToken);
            response.Data.Add(new EmbeddingEntry
            {
                Index = i,
                Embedding = vec,
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
}
