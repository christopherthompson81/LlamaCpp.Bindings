using LlamaCpp.Bindings.Server.Configuration;
using LlamaCpp.Bindings.Server.Endpoints;
using LlamaCpp.Bindings.Server.Models;
using LlamaCpp.Bindings.Server.Services;
using Microsoft.Extensions.Options;

// Minimal-API hosted llama.cpp server. V1 endpoints:
//   GET  /health                 — liveness probe
//   GET  /v1/models              — list the loaded model
//   POST /v1/chat/completions    — OpenAI-compatible chat (streaming or not)
//   POST /completion             — llama-server's native raw-text endpoint
//
// One model per process. Concurrent requests share the same context via
// LlamaSession leases, bounded by MaxSequenceCount; excess requests queue.

namespace LlamaCpp.Bindings.Server;

public class Program
{
    public static async Task<int> Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Bind configuration from appsettings.json + environment + CLI args.
        builder.Services
            .AddOptions<ServerOptions>()
            .Bind(builder.Configuration.GetSection(ServerOptions.Section))
            .ValidateOnStart();

        // Kestrel URLs are the one config value we need BEFORE Build (UseUrls
        // is a WebHost call). Read it straight from Configuration so it's
        // consistent with appsettings.json / CLI overrides. Everything else
        // reads after Build so test-time config hooks
        // (WebApplicationFactory.ConfigureAppConfiguration) actually apply.
        var urls = builder.Configuration[$"{ServerOptions.Section}:Urls"];
        if (!string.IsNullOrWhiteSpace(urls))
        {
            builder.WebHost.UseUrls(urls);
        }

        builder.Services.AddSingleton<ModelHost>();
        builder.Services.AddSingleton<SessionPool>();
        builder.Services.AddSingleton<EmbeddingHost>();

        var app = builder.Build();

        // Resolve the finalised options via IOptions — this picks up every
        // configuration source, including test-injected in-memory overrides.
        var serverOpts = app.Services.GetRequiredService<IOptions<ServerOptions>>().Value;

        // Eager-construct the model so any load failure surfaces at startup
        // rather than on the first request.
        var host = app.Services.GetRequiredService<ModelHost>();
        _ = app.Services.GetRequiredService<SessionPool>();
        // Eager-construct the embedding host too — its ctor is a no-op when
        // no embedding model path is configured, so this is free in the
        // default single-model deployment.
        _ = app.Services.GetRequiredService<EmbeddingHost>();

        // Resolve API keys once at startup from the inline list + optional
        // file. Any failure (missing file) throws here rather than on the
        // first request.
        var authLogger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("ApiKeyAuth");
        var validKeys = ApiKeyAuth.LoadKeys(serverOpts.ApiKeys, serverOpts.ApiKeyFile, authLogger);
        app.UseApiKeyAuth(validKeys);

        MapEndpoints(app);

        await app.RunAsync();
        return 0;
    }

    internal static void MapEndpoints(WebApplication app)
    {
        app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

        app.MapGet("/v1/models", (ModelHost host) =>
        {
            return new ModelsListResponse
            {
                Data = new()
                {
                    new ModelEntry { Id = host.ModelId, OwnedBy = "local" },
                },
            };
        });

        // Operator visibility into the session pool: which slots are
        // currently in use, how many cached tokens each holds, which
        // seq_id maps to which slot. Subject to the same API-key auth
        // gate as everything else — the snapshot reveals cache state
        // which could leak info about what other callers have asked.
        app.MapGet("/slots", (SessionPool pool) => pool.Snapshot());

        app.MapPost("/v1/chat/completions", async (
            HttpContext ctx,
            ChatCompletionsRequest req,
            ModelHost host,
            SessionPool pool,
            IOptions<ServerOptions> options,
            ILoggerFactory loggers,
            CancellationToken ct) =>
        {
            await ChatCompletionsEndpoint.Handle(ctx, req, host, pool, options, loggers, ct);
        });

        app.MapPost("/completion", async (
            HttpContext ctx,
            CompletionRequest req,
            ModelHost host,
            SessionPool pool,
            IOptions<ServerOptions> options,
            CancellationToken ct) =>
        {
            await CompletionEndpoint.Handle(ctx, req, host, pool, options, ct);
        });

        app.MapPost("/v1/embeddings", async (
            HttpContext ctx,
            EmbeddingsRequest req,
            EmbeddingHost embeddings,
            CancellationToken ct) =>
        {
            await EmbeddingsEndpoint.Handle(ctx, req, embeddings, ct);
        });
    }
}
