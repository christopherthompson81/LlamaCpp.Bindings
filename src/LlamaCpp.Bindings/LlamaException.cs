namespace LlamaCpp.Bindings;

/// <summary>
/// Thrown when a native llama.cpp call fails: either a NULL pointer return
/// from a constructor-like function, or a nonzero status code.
/// </summary>
public sealed class LlamaException : Exception
{
    /// <summary>Name of the native function that failed.</summary>
    public string FunctionName { get; }

    /// <summary>Nonzero status code returned by the native function, or null if the failure was a NULL pointer return.</summary>
    public int? StatusCode { get; }

    public LlamaException(string functionName, string message)
        : base(message)
    {
        FunctionName = functionName;
        StatusCode = null;
    }

    public LlamaException(string functionName, int statusCode)
        : base($"{functionName} returned status {statusCode}")
    {
        FunctionName = functionName;
        StatusCode = statusCode;
    }

    public LlamaException(string functionName, int statusCode, string message)
        : base(message)
    {
        FunctionName = functionName;
        StatusCode = statusCode;
    }
}
