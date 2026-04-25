using System.Security.Cryptography.X509Certificates;
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

        // Kestrel URLs + HTTPS cert + CORS origins are the config values we
        // need BEFORE Build (UseUrls / ConfigureKestrel / AddCors all have
        // to happen on the pre-Build builder). Everything else reads after
        // Build via IOptions so test-time overrides
        // (WebApplicationFactory.ConfigureAppConfiguration) take effect.
        var urls = builder.Configuration[$"{ServerOptions.Section}:Urls"];
        if (!string.IsNullOrWhiteSpace(urls))
        {
            builder.WebHost.UseUrls(urls);
        }

        var certPath = builder.Configuration[$"{ServerOptions.Section}:HttpsCertificatePath"];
        if (!string.IsNullOrWhiteSpace(certPath))
        {
            if (!File.Exists(certPath))
            {
                throw new FileNotFoundException(
                    $"LlamaServer:HttpsCertificatePath='{certPath}' but the file does not exist.", certPath);
            }
            var certPass = builder.Configuration[$"{ServerOptions.Section}:HttpsCertificatePassword"];
            // LoadPkcs12FromFile is the .NET 9+ replacement for the
            // X509Certificate2(path, password) ctor; keeps the obsoleted
            // ctor out of the build.
            var cert = X509CertificateLoader.LoadPkcs12FromFile(certPath, certPass);
            builder.WebHost.ConfigureKestrel(opts =>
                opts.ConfigureHttpsDefaults(https => https.ServerCertificate = cert));
        }

        // CORS service machinery is always registered — cheap, and we
        // can't know at this point whether the caller wants it (config
        // overrides from WebApplicationFactory.ConfigureAppConfiguration
        // aren't visible until Build finalises the config pipeline).
        // The actual policy is attached below via app.UseCors, by which
        // time we've resolved ServerOptions via IOptions.
        builder.Services.AddCors();

        builder.Services.AddSingleton<ModelHost>();
        builder.Services.AddSingleton<SessionPool>();
        builder.Services.AddSingleton<EmbeddingHost>();
        builder.Services.AddSingleton<RerankHost>();
        builder.Services.AddSingleton<MmprojHost>();
        builder.Services.AddSingleton<ServerMetrics>();

        // Structured HTTP request logging — {method} {path} {status}
        // {duration} for every request. The default level is Information
        // so operators can silence it via the usual Logging config.
        builder.Services.AddHttpLogging(opts =>
        {
            opts.LoggingFields =
                Microsoft.AspNetCore.HttpLogging.HttpLoggingFields.RequestMethod |
                Microsoft.AspNetCore.HttpLogging.HttpLoggingFields.RequestPath |
                Microsoft.AspNetCore.HttpLogging.HttpLoggingFields.ResponseStatusCode |
                Microsoft.AspNetCore.HttpLogging.HttpLoggingFields.Duration;
        });

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
        _ = app.Services.GetRequiredService<RerankHost>();
        _ = app.Services.GetRequiredService<MmprojHost>();

        // HTTP request logging + per-endpoint request counter. Both sit
        // at the top of the pipeline so they see everything including
        // 401s, preflights, and malformed requests. The counter is a
        // shared singleton (ServerMetrics) feeding /metrics.
        app.UseHttpLogging();
        app.Use(async (ctx, next) =>
        {
            await next();
            var metrics = ctx.RequestServices.GetRequiredService<ServerMetrics>();
            metrics.IncrementRequest(ctx.Request.Path.ToString(), ctx.Response.StatusCode);
        });

        // CORS runs BEFORE auth: preflight OPTIONS requests from browsers
        // don't carry the Authorization header, so they'd otherwise 401
        // and the actual request would never get sent.
        if (serverOpts.CorsAllowedOrigins is { Count: > 0 } corsOrigins)
        {
            app.UseCors(policy =>
            {
                policy.AllowAnyHeader().AllowAnyMethod();
                bool wildcard = corsOrigins.Contains("*");
                // Per the CORS spec, Access-Control-Allow-Origin: * is
                // incompatible with Access-Control-Allow-Credentials: true.
                // When both are requested we mirror the incoming Origin
                // instead — same permissive effect, spec-compliant.
                if (wildcard && !serverOpts.CorsAllowCredentials)
                {
                    policy.AllowAnyOrigin();
                }
                else if (wildcard)
                {
                    policy.SetIsOriginAllowed(_ => true);
                }
                else
                {
                    policy.WithOrigins(corsOrigins.ToArray());
                }
                if (serverOpts.CorsAllowCredentials)
                {
                    policy.AllowCredentials();
                }
            });
        }

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

        // Prometheus scrape target. Counters are incremented by the
        // request-logging middleware + the generator loops; slot gauges
        // are snapshotted from the pool on each scrape.
        app.MapGet("/metrics", (ServerMetrics metrics, SessionPool pool) =>
            Results.Content(metrics.Render(pool), "text/plain; version=0.0.4; charset=utf-8"));

        app.MapPost("/v1/chat/completions", async (
            HttpContext ctx,
            ChatCompletionsRequest req,
            ModelHost host,
            SessionPool pool,
            IOptions<ServerOptions> options,
            ILoggerFactory loggers,
            ServerMetrics metrics,
            MmprojHost mmproj,
            CancellationToken ct) =>
        {
            await ChatCompletionsEndpoint.Handle(ctx, req, host, pool, options, loggers, metrics, mmproj, ct);
        });

        app.MapPost("/completion", async (
            HttpContext ctx,
            CompletionRequest req,
            ModelHost host,
            SessionPool pool,
            IOptions<ServerOptions> options,
            ServerMetrics metrics,
            CancellationToken ct) =>
        {
            await CompletionEndpoint.Handle(ctx, req, host, pool, options, metrics, ct);
        });

        app.MapPost("/v1/embeddings", async (
            HttpContext ctx,
            EmbeddingsRequest req,
            EmbeddingHost embeddings,
            CancellationToken ct) =>
        {
            await EmbeddingsEndpoint.Handle(ctx, req, embeddings, ct);
        });

        app.MapPost("/v1/rerank", async (
            HttpContext ctx,
            RerankRequest req,
            RerankHost rerank,
            CancellationToken ct) =>
        {
            await RerankEndpoint.Handle(ctx, req, rerank, ct);
        });
    }
}
