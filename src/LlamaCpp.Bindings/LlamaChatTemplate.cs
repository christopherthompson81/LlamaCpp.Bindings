using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Text;
using LlamaCpp.Bindings.Jinja;
using LlamaCpp.Bindings.Native;

namespace LlamaCpp.Bindings;

/// <summary>
/// A single message in a chat conversation. <paramref name="Role"/> is
/// typically <c>"system"</c>, <c>"user"</c>, or <c>"assistant"</c> but the
/// template ultimately decides what's valid — llama.cpp passes the role
/// through to the Jinja template unchanged.
/// </summary>
public readonly record struct ChatMessage(string Role, string Content);

/// <summary>
/// Renders a chat conversation to a prompt string using a Jinja template.
/// Wrapper for <c>llama_chat_apply_template</c>. Intended usage is:
/// <code>
/// var tmpl = model.GetChatTemplate() ?? throw ...;
/// var prompt = LlamaChatTemplate.Apply(tmpl, messages);
/// var tokens = model.Vocab.Tokenize(prompt, addSpecial: false, parseSpecial: true);
/// </code>
/// </summary>
/// <remarks>
/// llama.cpp's chat template applier is **not** a full Jinja engine — it
/// supports a curated set of known templates (Llama 2/3, ChatML, Mistral,
/// Gemma, Qwen, etc.). If a model ships an exotic template that isn't
/// recognised, this throws <see cref="LlamaException"/>.
/// </remarks>
public static class LlamaChatTemplate
{
    /// <summary>
    /// Names of chat templates llama.cpp can apply by name (Llama 2/3, ChatML,
    /// Mistral, Gemma, Qwen, etc.). Pass one of these to
    /// <see cref="Apply"/> as the template, or use a model's own embedded
    /// template via <see cref="LlamaModel.GetChatTemplate"/>.
    /// </summary>
    public static IReadOnlyList<string> BuiltInTemplateNames()
    {
        LlamaBackend.EnsureInitialized();

        // First call with len=0 returns the number of templates available.
        int count;
        unsafe
        {
            count = NativeMethods.llama_chat_builtin_templates(null, 0);
        }
        if (count <= 0) return Array.Empty<string>();

        var ptrs = new IntPtr[count];
        unsafe
        {
            fixed (IntPtr* buf = ptrs)
            {
                int written = NativeMethods.llama_chat_builtin_templates(buf, (nuint)count);
                if (written < 0)
                {
                    throw new LlamaException(
                        nameof(NativeMethods.llama_chat_builtin_templates), written,
                        "llama_chat_builtin_templates failed.");
                }
            }
        }
        var names = new string[count];
        for (int i = 0; i < count; i++)
        {
            names[i] = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(ptrs[i]) ?? string.Empty;
        }
        return names;
    }

    /// <summary>
    /// Apply a chat template to a conversation.
    /// </summary>
    /// <param name="template">
    /// A Jinja-style template string — typically obtained from
    /// <see cref="LlamaModel.GetChatTemplate"/>.
    /// </param>
    /// <param name="messages">Ordered list of chat messages.</param>
    /// <param name="addAssistantPrefix">
    /// If true, the rendered prompt ends with the template's assistant-turn
    /// prefix, ready for the model to generate the reply. Use this when
    /// generating; disable when serialising a complete multi-turn transcript.
    /// </param>
    public static string Apply(
        string template,
        IReadOnlyList<ChatMessage> messages,
        bool addAssistantPrefix = true,
        IReadOnlyList<object?>? tools = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(template);
        ArgumentNullException.ThrowIfNull(messages);
        LlamaBackend.EnsureInitialized();

        if (messages.Count == 0)
        {
            throw new ArgumentException("Chat template needs at least one message.", nameof(messages));
        }

        // Prefer our own Jinja interpreter — it handles the Qwen3-family,
        // DeepSeek, GLM, Kimi, etc. templates that llama.cpp's curated-matcher
        // path mis-applies. Native fallback is retained for edge cases we
        // haven't covered yet (e.g. exotic macros / inheritance).
        try
        {
            return ApplyWithJinja(template, messages, addAssistantPrefix, tools);
        }
        catch (JinjaException)
        {
            // Fall through to the native curated-matcher path. Swallowing
            // silently is intentional — unusual templates that confuse us
            // still often match one of llama.cpp's built-in recognisers.
        }

        // Marshal every role/content as a UTF-8 null-terminated block, then
        // point llama_chat_message.role/content at them. The lifetimes must
        // strictly cover the native call; we release in the finally.
        var handles = new IntPtr[messages.Count * 2]; // [role0, content0, role1, content1, ...]
        var chat = new llama_chat_message[messages.Count];

        int totalChars = 0;
        for (int i = 0; i < messages.Count; i++)
        {
            var m = messages[i];
            if (m.Role is null || m.Content is null)
            {
                throw new ArgumentException($"Chat message {i} has null Role or Content.", nameof(messages));
            }
            totalChars += m.Role.Length + m.Content.Length;
        }

        try
        {
            for (int i = 0; i < messages.Count; i++)
            {
                var roleH    = Marshal.StringToCoTaskMemUTF8(messages[i].Role);
                var contentH = Marshal.StringToCoTaskMemUTF8(messages[i].Content);
                handles[i * 2]     = roleH;
                handles[i * 2 + 1] = contentH;
                chat[i] = new llama_chat_message { role = roleH, content = contentH };
            }

            return ApplyTemplateCore(template, chat, addAssistantPrefix, totalChars);
        }
        finally
        {
            for (int i = 0; i < handles.Length; i++)
            {
                if (handles[i] != IntPtr.Zero) Marshal.FreeCoTaskMem(handles[i]);
            }
        }
    }

    /// <summary>
    /// Compiled-template cache keyed by the raw template source. Model
    /// templates are stable for the lifetime of a loaded model, so this is
    /// effectively one entry per loaded model. Thread-safe by construction.
    /// </summary>
    private static readonly ConcurrentDictionary<string, JinjaTemplate> _jinjaCache = new();

    private static string ApplyWithJinja(
        string template,
        IReadOnlyList<ChatMessage> messages,
        bool addAssistantPrefix,
        IReadOnlyList<object?>? tools = null)
    {
        var compiled = _jinjaCache.GetOrAdd(template, JinjaTemplate.Parse);

        var msgList = new List<object?>(messages.Count);
        foreach (var m in messages)
        {
            msgList.Add(new Dictionary<string, object?>
            {
                ["role"] = m.Role,
                ["content"] = m.Content,
            });
        }

        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = msgList,
            ["add_generation_prompt"] = addAssistantPrefix,
            // Conservative defaults for flags some templates branch on.
            // Keeping enable_thinking unset lets the Qwen3 thinking-mode
            // branch render (its default is "thinking on").
            ["tools"] = tools is { Count: > 0 } ? (object?)tools : null,
            ["add_vision_id"] = false,
        };

        return compiled.Render(ctx);
    }

    private static unsafe string ApplyTemplateCore(
        string template,
        llama_chat_message[] chat,
        bool addAssistantPrefix,
        int totalCharsHint)
    {
        // Header comment on llama_chat_apply_template suggests 2x total chars
        // as a starting allocation. Floor it so trivial prompts still get a
        // reasonable first shot.
        int cap = Math.Max(256, totalCharsHint * 2);
        var buf = new byte[cap];

        int written;
        fixed (llama_chat_message* chatPtr = chat)
        fixed (byte* bufPtr = buf)
        {
            written = NativeMethods.llama_chat_apply_template(
                template, chatPtr, (nuint)chat.Length,
                addAssistantPrefix, bufPtr, buf.Length);
        }

        if (written < 0)
        {
            throw new LlamaException(
                nameof(NativeMethods.llama_chat_apply_template),
                written,
                "llama_chat_apply_template rejected this template/message combination. " +
                "The applier supports only the curated set of built-in templates — unusual " +
                "Jinja templates from GGUF metadata may not work.");
        }

        if (written > buf.Length)
        {
            // Buffer too small on first try — realloc to exactly the reported
            // size and retry. If it still reports more, something is wrong.
            buf = new byte[written];
            fixed (llama_chat_message* chatPtr = chat)
            fixed (byte* bufPtr = buf)
            {
                written = NativeMethods.llama_chat_apply_template(
                    template, chatPtr, (nuint)chat.Length,
                    addAssistantPrefix, bufPtr, buf.Length);
            }
            if (written < 0 || written > buf.Length)
            {
                throw new LlamaException(
                    nameof(NativeMethods.llama_chat_apply_template),
                    written,
                    "Chat template rendering unstable across retries.");
            }
        }

        return Encoding.UTF8.GetString(buf, 0, written);
    }
}
