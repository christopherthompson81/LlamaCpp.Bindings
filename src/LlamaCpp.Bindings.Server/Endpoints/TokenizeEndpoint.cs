using System.Text.Json.Serialization;
using LlamaCpp.Bindings.Server.Services;
using Microsoft.AspNetCore.Http;

namespace LlamaCpp.Bindings.Server.Endpoints;

/// <summary>
/// llama-server's <c>POST /tokenize</c>: turn text into token IDs using
/// the loaded model's vocab. Optionally returns the per-token "piece"
/// string for clients that want a token-by-text breakdown (UI displays,
/// debug tools).
/// </summary>
public static class TokenizeEndpoint
{
    public static async Task Handle(HttpContext http, TokenizeRequest req, ModelHost host, CancellationToken ct)
    {
        if (req.Content is null)
        {
            // Upstream returns an empty token list when content is missing;
            // we mirror that rather than 400 so existing scripts don't break.
            await http.Response.WriteAsJsonAsync(
                new TokenizeResponse { Tokens = Array.Empty<int>() }, ct);
            return;
        }

        var ids = host.Model.Vocab.Tokenize(req.Content, req.AddSpecial ?? false, req.ParseSpecial ?? true);
        if (req.WithPieces == true)
        {
            var pieces = new List<TokenPiece>(ids.Length);
            foreach (var id in ids)
            {
                pieces.Add(new TokenPiece
                {
                    Id = id,
                    // renderSpecial=true so callers see the actual rendered
                    // text for special tokens — matches upstream behavior.
                    Piece = host.Model.Vocab.TokenToPiece(id, renderSpecial: true),
                });
            }
            await http.Response.WriteAsJsonAsync(new TokenizeResponseWithPieces { Tokens = pieces }, ct);
            return;
        }

        await http.Response.WriteAsJsonAsync(new TokenizeResponse { Tokens = ids }, ct);
    }
}

/// <summary>
/// llama-server's <c>POST /detokenize</c>: turn token IDs back into text.
/// </summary>
public static class DetokenizeEndpoint
{
    public static async Task Handle(HttpContext http, DetokenizeRequest req, ModelHost host, CancellationToken ct)
    {
        if (req.Tokens is null || req.Tokens.Length == 0)
        {
            await http.Response.WriteAsJsonAsync(new DetokenizeResponse { Content = "" }, ct);
            return;
        }

        // Concatenate piece text. Upstream's tokens_to_str uses
        // common_detokenize, which for our purposes is equivalent to
        // walking each id and joining its piece (TokenToPiece already
        // handles the leading-space / byte-fallback edge cases).
        var sb = new System.Text.StringBuilder();
        foreach (var id in req.Tokens)
        {
            sb.Append(host.Model.Vocab.TokenToPiece(id, renderSpecial: true));
        }
        await http.Response.WriteAsJsonAsync(new DetokenizeResponse { Content = sb.ToString() }, ct);
    }
}

public sealed class TokenizeRequest
{
    [JsonPropertyName("content")]
    public string? Content { get; set; }

    /// <summary>llama.cpp's <c>add_special</c> — prepend BOS/special tokens. Default false (matches upstream).</summary>
    [JsonPropertyName("add_special")]
    public bool? AddSpecial { get; set; }

    /// <summary>llama.cpp's <c>parse_special</c> — recognise special-token markup in the input. Default true.</summary>
    [JsonPropertyName("parse_special")]
    public bool? ParseSpecial { get; set; }

    /// <summary>When true, response is an array of <see cref="TokenPiece"/> objects instead of bare integers.</summary>
    [JsonPropertyName("with_pieces")]
    public bool? WithPieces { get; set; }
}

public sealed class TokenizeResponse
{
    [JsonPropertyName("tokens")]
    public int[] Tokens { get; set; } = Array.Empty<int>();
}

public sealed class TokenizeResponseWithPieces
{
    [JsonPropertyName("tokens")]
    public List<TokenPiece> Tokens { get; set; } = new();
}

public sealed class TokenPiece
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("piece")]
    public string Piece { get; set; } = "";
}

public sealed class DetokenizeRequest
{
    [JsonPropertyName("tokens")]
    public int[]? Tokens { get; set; }
}

public sealed class DetokenizeResponse
{
    [JsonPropertyName("content")]
    public string Content { get; set; } = "";
}
