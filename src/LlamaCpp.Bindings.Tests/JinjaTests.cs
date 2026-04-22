using LlamaCpp.Bindings.Jinja;

namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Unit tests for the Jinja subset interpreter. No model load required — these
/// run against pure string input/output and cover the constructs real chat
/// templates use.
/// </summary>
public class JinjaTests
{
    private static string Render(string source, Dictionary<string, object?>? ctx = null) =>
        JinjaTemplate.Parse(source).Render(ctx ?? new());

    // ========== Basics ==========

    [Fact]
    public void Plain_Text_Passes_Through() =>
        Assert.Equal("hello", Render("hello"));

    [Fact]
    public void Output_Expression_Renders_Variable() =>
        Assert.Equal("hi Chris", Render("hi {{ name }}", new() { ["name"] = "Chris" }));

    [Fact]
    public void Undefined_Variable_Renders_Empty() =>
        Assert.Equal("x=[]", Render("x=[{{ missing }}]"));

    [Fact]
    public void Whitespace_Strip_Respects_Minus_Flags()
    {
        var src = "A\n{%- if true -%}\n  B\n{%- endif -%}\nC";
        Assert.Equal("ABC", Render(src));
    }

    [Fact]
    public void Comment_Is_Stripped() =>
        Assert.Equal("ab", Render("a{# not rendered #}b"));

    // ========== Expressions ==========

    [Fact]
    public void String_Concat_With_Tilde() =>
        Assert.Equal("HelloWorld",
            Render("{{ 'Hello' ~ 'World' }}"));

    [Fact]
    public void String_Concat_With_Plus() =>
        Assert.Equal("ab",
            Render("{{ 'a' + 'b' }}"));

    [Fact]
    public void Arithmetic_Operators()
    {
        Assert.Equal("5", Render("{{ 2 + 3 }}"));
        Assert.Equal("6", Render("{{ 2 * 3 }}"));
        Assert.Equal("2", Render("{{ 7 // 3 }}"));
        Assert.Equal("1", Render("{{ 7 % 3 }}"));
    }

    [Fact]
    public void Comparisons_And_Logic()
    {
        Assert.Equal("True", Render("{{ 1 < 2 and 2 < 3 }}"));
        Assert.Equal("False", Render("{{ 1 < 2 and 3 < 2 }}"));
        Assert.Equal("True", Render("{{ 1 == 1 or 'x' == 'y' }}"));
        Assert.Equal("False", Render("{{ not (1 == 1) }}"));
    }

    [Fact]
    public void In_And_NotIn()
    {
        Assert.Equal("True", Render("{{ 'a' in 'abc' }}"));
        Assert.Equal("False", Render("{{ 'd' in 'abc' }}"));
        Assert.Equal("True", Render("{{ 'b' not in 'ac' }}"));
    }

    [Fact]
    public void Ternary()
    {
        Assert.Equal("yes", Render("{{ 'yes' if flag else 'no' }}", new() { ["flag"] = true }));
        Assert.Equal("no", Render("{{ 'yes' if flag else 'no' }}", new() { ["flag"] = false }));
    }

    // ========== Member / index / slice ==========

    [Fact]
    public void Member_Access_On_Dict()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["user"] = new Dictionary<string, object?> { ["name"] = "Alice" }
        };
        Assert.Equal("Alice", Render("{{ user.name }}", ctx));
    }

    [Fact]
    public void Index_Access_On_List_And_Dict()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { "a", "b", "c" },
            ["m"] = new Dictionary<string, object?> { ["k"] = "v" },
        };
        Assert.Equal("a", Render("{{ xs[0] }}", ctx));
        Assert.Equal("c", Render("{{ xs[-1] }}", ctx));
        Assert.Equal("v", Render("{{ m['k'] }}", ctx));
    }

    [Fact]
    public void Slice_Forward()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { "a", "b", "c", "d" },
        };
        Assert.Equal("[b, c]", Render("{{ xs[1:3] }}", ctx));
    }

    [Fact]
    public void Slice_Reverse_Step()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { "a", "b", "c" },
        };
        Assert.Equal("[c, b, a]", Render("{{ xs[::-1] }}", ctx));
    }

    // ========== Statements ==========

    [Fact]
    public void If_Elif_Else()
    {
        var tmpl = "{% if x == 1 %}one{% elif x == 2 %}two{% else %}many{% endif %}";
        Assert.Equal("one", Render(tmpl, new() { ["x"] = 1L }));
        Assert.Equal("two", Render(tmpl, new() { ["x"] = 2L }));
        Assert.Equal("many", Render(tmpl, new() { ["x"] = 9L }));
    }

    [Fact]
    public void For_Loop_With_Index_And_First_Last()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { "a", "b", "c" },
        };
        var src = "{% for x in xs %}{{ loop.index0 }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}";
        Assert.Equal("0:a,1:b,2:c", Render(src, ctx));
    }

    [Fact]
    public void For_With_Previtem_And_Nextitem()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { "a", "b", "c" },
        };
        // previtem on first is undefined→empty; nextitem on last same.
        var src = "{% for x in xs %}[{{ loop.previtem }}|{{ x }}|{{ loop.nextitem }}]{% endfor %}";
        Assert.Equal("[|a|b][a|b|c][b|c|]", Render(src, ctx));
    }

    [Fact]
    public void Set_Assignment_Local() =>
        Assert.Equal("42", Render("{% set x = 42 %}{{ x }}"));

    [Fact]
    public void Namespace_Mutates_Across_Loop_Iterations()
    {
        // Inside a for-loop, `set ns.x = ...` persists — that's the whole
        // point of namespace(). Plain `set` would be scoped per-iteration.
        var src = "{% set ns = namespace(total=0) %}" +
                  "{% for n in xs %}{% set ns.total = ns.total + n %}{% endfor %}" +
                  "{{ ns.total }}";
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { 1L, 2L, 3L, 4L },
        };
        Assert.Equal("10", Render(src, ctx));
    }

    // ========== Tests ==========

    [Fact]
    public void Is_Defined_Is_None_Is_String()
    {
        Assert.Equal("True", Render("{{ x is defined }}", new() { ["x"] = 1L }));
        Assert.Equal("False", Render("{{ x is defined }}"));
        Assert.Equal("True", Render("{{ x is none }}", new() { ["x"] = null }));
        Assert.Equal("True", Render("{{ x is string }}", new() { ["x"] = "hi" }));
        Assert.Equal("True",
            Render("{{ x is not none }}", new() { ["x"] = "hi" }));
    }

    // ========== Filters ==========

    [Fact]
    public void Filter_Trim_Length_Upper()
    {
        Assert.Equal("hi", Render("{{ '  hi  ' | trim }}"));
        Assert.Equal("5", Render("{{ 'hello' | length }}"));
        Assert.Equal("HI", Render("{{ 'hi' | upper }}"));
    }

    [Fact]
    public void Filter_Chain_Is_Left_To_Right() =>
        Assert.Equal("ABC", Render("{{ '  abc  ' | trim | upper }}"));

    [Fact]
    public void Filter_Tojson()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["x"] = new Dictionary<string, object?> { ["a"] = 1L, ["b"] = "two" },
        };
        var result = Render("{{ x | tojson }}", ctx);
        // JSON key order matches insertion; both orderings are acceptable
        // depending on dict impl, so accept either.
        Assert.True(result == "{\"a\":1,\"b\":\"two\"}" || result == "{\"b\":\"two\",\"a\":1}");
    }

    // ========== String methods ==========

    [Fact]
    public void String_Method_Calls()
    {
        Assert.Equal("True", Render("{{ 'abcde'.startswith('abc') }}"));
        Assert.Equal("True", Render("{{ 'abcde'.endswith('de') }}"));
        Assert.Equal("bc", Render("{{ '  bc  '.strip() }}"));
        Assert.Equal("foo_bar", Render("{{ 'foo.bar'.replace('.', '_') }}"));
        // split returns a list; indexing picks one element
        Assert.Equal("tail", Render("{{ 'head-tail'.split('-')[-1] }}"));
    }

    // ========== Macros ==========

    [Fact]
    public void Macro_Definition_And_Invocation()
    {
        var src = "{% macro greet(name, excited=false) -%}" +
                  "Hello {{ name }}{% if excited %}!{% endif %}" +
                  "{%- endmacro %}" +
                  "{{ greet('world') }} / {{ greet('world', true) }}";
        Assert.Equal("Hello world / Hello world!", Render(src));
    }

    [Fact]
    public void Raise_Exception_Bubbles_Up()
    {
        var ex = Assert.Throws<JinjaException>(() =>
            Render("{{ raise_exception('boom') }}"));
        Assert.Contains("boom", ex.Message);
    }

    // ========== Qwen3.6-style integration ==========

    [Fact]
    public void Qwen36_Minimal_SingleTurn_Render()
    {
        // A stripped-down version of the Qwen3.6 template focusing on the
        // plumbing that was failing under llama_chat_apply_template: for
        // loop, message.role dispatch, string concat, and the ending
        // <think>\n generation prefix.
        var tmpl = """
            {%- for message in messages %}
                {%- if message.role == "user" -%}
                    <|im_start|>{{ message.role }}
            {{ message.content }}<|im_end|>
            {% elif message.role == "assistant" -%}
                    <|im_start|>{{ message.role }}
            {{ message.content }}<|im_end|>
            {% endif -%}
            {% endfor -%}
            {%- if add_generation_prompt -%}
            <|im_start|>assistant
            <think>
            {%- endif %}
            """;
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?>
                {
                    ["role"] = "user",
                    ["content"] = "Hello, could you introduce yourself?",
                },
            },
            ["add_generation_prompt"] = true,
        };

        var result = Render(tmpl, ctx);

        Assert.Contains("<|im_start|>user", result);
        Assert.Contains("Hello, could you introduce yourself?", result);
        Assert.Contains("<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n<think>", result);
    }

    [Fact]
    public void Qwen36_Full_Template_Renders_SingleTurn()
    {
        // The actual template dumped from a Qwen3.6-35B-A3B GGUF (staged under
        // TestData). Acceptance test: parses without error, and for a
        // single-turn "hello" the output ends with
        // <|im_start|>assistant\n<think>\n — which is the exact behaviour the
        // chatml fallback path *wasn't* producing, and the root cause of the
        // 1/3 off-topic wander.
        var path = System.IO.Path.Combine(
            System.IO.Path.GetDirectoryName(typeof(JinjaTests).Assembly.Location)!,
            "TestData", "qwen36-template.jinja");
        if (!System.IO.File.Exists(path))
        {
            Console.WriteLine($"SKIP: no Qwen3.6 template at {path}.");
            return;
        }

        var source = System.IO.File.ReadAllText(path);
        var tmpl = JinjaTemplate.Parse(source);

        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?>
                {
                    ["role"] = "user",
                    ["content"] = "Hello, could you introduce yourself?",
                },
            },
            ["add_generation_prompt"] = true,
        };

        var result = tmpl.Render(ctx);

        Assert.Contains("<|im_start|>user", result);
        Assert.Contains("Hello, could you introduce yourself?", result);
        Assert.Contains("<|im_end|>", result);
        // The crucial assertion: the thinking-mode block MUST be pre-opened.
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", result);
    }

    [Fact]
    public void Qwen36_Reverse_Loop_Finds_Last_User_Index()
    {
        // The Qwen3.6 template runs a reverse-loop over messages to find the
        // last non-tool-response user message (for thinking-mode placement).
        // Verify the reverse-slice + `set ns.x` combination we need.
        var src = """
            {%- set ns = namespace(last_user_index=-1) %}
            {%- for message in messages[::-1] %}
                {%- set index = (messages|length - 1) - loop.index0 %}
                {%- if message.role == "user" and ns.last_user_index == -1 %}
                    {%- set ns.last_user_index = index %}
                {%- endif %}
            {%- endfor %}
            {{ ns.last_user_index }}
            """;
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?> { ["role"] = "system", ["content"] = "..." },
                new Dictionary<string, object?> { ["role"] = "user", ["content"] = "q1" },
                new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "a1" },
                new Dictionary<string, object?> { ["role"] = "user", ["content"] = "q2" },
            },
        };

        var result = Render(src, ctx).Trim();
        Assert.Equal("3", result);
    }
}
