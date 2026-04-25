using LlamaCpp.Bindings.Server.Models;
using LlamaCpp.Bindings.Server.Services;
using Microsoft.AspNetCore.Http;

namespace LlamaCpp.Bindings.Server.Endpoints;

/// <summary>
/// <c>POST /v1/rerank</c> — Cohere/Jina-style reranking endpoint.
/// Scores each document against the query, sorts by descending
/// relevance, optionally truncates to <c>top_n</c>, and returns the
/// list. When the server isn't configured with a reranker model the
/// endpoint returns 501 (same shape as <c>/v1/embeddings</c> when
/// embeddings aren't configured).
/// </summary>
public static class RerankEndpoint
{
    public static async Task Handle(
        HttpContext http,
        RerankRequest req,
        RerankHost rerank,
        CancellationToken cancellationToken)
    {
        if (!rerank.IsAvailable)
        {
            http.Response.StatusCode = StatusCodes.Status501NotImplemented;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message = "This server is not configured with a rerank model. " +
                              "Set LlamaServer:RerankModelPath to enable /v1/rerank.",
                    type = "not_implemented",
                    code = "rerank_not_configured",
                },
            }, cancellationToken);
            return;
        }

        if (string.IsNullOrEmpty(req.Query))
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new { message = "query must not be empty", type = "invalid_request_error" },
            }, cancellationToken);
            return;
        }
        if (req.Documents is null || req.Documents.Count == 0)
        {
            http.Response.StatusCode = StatusCodes.Status400BadRequest;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new { message = "documents must not be empty", type = "invalid_request_error" },
            }, cancellationToken);
            return;
        }

        // Score every document. Sequential — RerankHost serialises calls
        // through a single semaphore. Batched scoring would help here
        // (issue #16-style work for the rerank context); deferred.
        var scores = new float[req.Documents.Count];
        try
        {
            for (int i = 0; i < req.Documents.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                scores[i] = await rerank.ScoreAsync(req.Query, req.Documents[i] ?? "", cancellationToken);
            }
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            // Surface the underlying error in the response body so
            // misconfigured rerank models (wrong pooling head, missing
            // encoder, etc.) are debuggable from the wire.
            http.Response.StatusCode = StatusCodes.Status500InternalServerError;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new { message = ex.Message, type = ex.GetType().Name },
            }, cancellationToken);
            return;
        }

        // Pair indices with their scores, sort descending by score, then
        // optionally trim to top_n.
        var ranked = scores
            .Select((score, idx) => new { Index = idx, Score = score })
            .OrderByDescending(x => x.Score)
            .ToArray();

        int take = req.TopN is int n && n > 0 ? Math.Min(n, ranked.Length) : ranked.Length;
        bool echo = req.ReturnDocuments == true;

        var response = new RerankResponse
        {
            Model = rerank.ModelId ?? "",
            Results = new List<RerankResult>(take),
        };
        for (int i = 0; i < take; i++)
        {
            var entry = ranked[i];
            response.Results.Add(new RerankResult
            {
                Index = entry.Index,
                RelevanceScore = entry.Score,
                Document = echo ? req.Documents[entry.Index] : null,
            });
        }

        http.Response.ContentType = "application/json";
        await http.Response.WriteAsJsonAsync(response, cancellationToken: cancellationToken);
    }
}
