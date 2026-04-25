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

        var serverOpts = builder.Configuration.GetSection(ServerOptions.Section).Get<ServerOptions>()
                         ?? new ServerOptions();

        // Kestrel URLs — take the value from options so CLI overrides apply.
        if (!string.IsNullOrWhiteSpace(serverOpts.Urls))
        {
            builder.WebHost.UseUrls(serverOpts.Urls);
        }

        builder.Services.AddSingleton<ModelHost>();
        builder.Services.AddSingleton<SessionPool>();

        var app = builder.Build();

        // Eager-construct the model so any load failure surfaces at startup
        // rather than on the first request.
        var host = app.Services.GetRequiredService<ModelHost>();
        _ = app.Services.GetRequiredService<SessionPool>();

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
    }
}
