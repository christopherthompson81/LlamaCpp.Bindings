using System;
using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Services.Mermaid;

/// <summary>
/// Parse a subset of Mermaid flowchart syntax:
/// <list type="bullet">
///   <item>Optional header <c>graph &lt;DIR&gt;</c> or <c>flowchart &lt;DIR&gt;</c> (TD/TB/BT/LR/RL). Defaults to TD.</item>
///   <item>Node shapes: <c>A[rect]</c>, <c>A(rounded)</c>, <c>A([stadium])</c>, <c>A((circle))</c>, <c>A{rhombus}</c>.</item>
///   <item>Edges: <c>--&gt;</c>, <c>---</c>, <c>-.-&gt;</c>, <c>==&gt;</c>.</item>
///   <item>Optional pipe-form edge labels: <c>A --&gt;|text| B</c>.</item>
///   <item>Chains: <c>A --&gt; B --&gt; C</c> produces two edges.</item>
///   <item>Line comments starting with <c>%%</c>.</item>
/// </list>
/// Not supported in v1: subgraphs, <c>A &amp; B</c> shorthand, class defs,
/// click handlers, linkStyle, interactions, direction overrides inside subgraphs.
/// Unknown lines are silently dropped — the goal is robustness to LLM drift.
/// </summary>
public static class FlowchartParser
{
    public static FlowchartGraph Parse(string source)
    {
        var g = new FlowchartGraph();
        if (string.IsNullOrWhiteSpace(source)) return g;

        var lines = source.Replace("\r\n", "\n").Split('\n');
        var headerSeen = false;

        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0) continue;
            if (line.StartsWith("%%")) continue;  // comment

            if (!headerSeen && TryParseHeader(line, g))
            {
                headerSeen = true;
                continue;
            }

            // Ignored keywords we don't support yet — swallow quietly.
            if (line.StartsWith("subgraph ", StringComparison.Ordinal) || line == "end"
                || line.StartsWith("classDef ", StringComparison.Ordinal)
                || line.StartsWith("class ", StringComparison.Ordinal)
                || line.StartsWith("click ", StringComparison.Ordinal)
                || line.StartsWith("linkStyle ", StringComparison.Ordinal)
                || line.StartsWith("style ", StringComparison.Ordinal))
            {
                continue;
            }

            ParseStatement(line, g);
        }

        return g;
    }

    private static bool TryParseHeader(string line, FlowchartGraph g)
    {
        string? rest = null;
        if (line.StartsWith("graph ", StringComparison.Ordinal)) rest = line.Substring(6);
        else if (line.StartsWith("flowchart ", StringComparison.Ordinal)) rest = line.Substring(10);
        else if (line == "graph" || line == "flowchart") { return true; }  // no direction → default TD
        if (rest is null) return false;

        rest = rest.Trim();
        g.Direction = rest switch
        {
            "TD" or "TB" => FlowchartDirection.TopDown,
            "BT" => FlowchartDirection.BottomUp,
            "LR" => FlowchartDirection.LeftRight,
            "RL" => FlowchartDirection.RightLeft,
            _ => FlowchartDirection.TopDown,
        };
        return true;
    }

    /// <summary>
    /// Parse one statement line. May be a single node declaration or a chain
    /// of node references separated by edges.
    /// </summary>
    private static void ParseStatement(string line, FlowchartGraph g)
    {
        var i = 0;
        SkipWs(line, ref i);
        if (i >= line.Length) return;

        var firstNode = TryParseNodeRef(line, ref i, g);
        if (firstNode is null) return;

        var currentFromId = firstNode;

        while (true)
        {
            SkipWs(line, ref i);
            if (i >= line.Length) return;

            if (!TryParseEdge(line, ref i, out var style, out var edgeLabel)) return;

            SkipWs(line, ref i);
            var nextNode = TryParseNodeRef(line, ref i, g);
            if (nextNode is null) return;

            g.Edges.Add(new FlowchartEdge(currentFromId, nextNode, style, edgeLabel));
            currentFromId = nextNode;
        }
    }

    /// <summary>
    /// Parse an id with optional shape+label; insert into graph if not already
    /// present, and return the id. Returns null on parse failure.
    /// </summary>
    private static string? TryParseNodeRef(string line, ref int i, FlowchartGraph g)
    {
        var id = ParseIdent(line, ref i);
        if (id is null) return null;

        // Optional shape body directly after the id (no whitespace).
        string? label = null;
        NodeShape shape = NodeShape.Rectangle;
        if (i < line.Length && TryParseShape(line, ref i, out var bodyLabel, out var bodyShape))
        {
            label = bodyLabel;
            shape = bodyShape;
        }

        // Only add if new id, or replace default label-less entry once a
        // labelled occurrence shows up.
        AddOrUpdateNode(g, id, label ?? id, shape, bodyLabelSeen: label is not null);
        return id;
    }

    private static void AddOrUpdateNode(FlowchartGraph g, string id, string label, NodeShape shape, bool bodyLabelSeen)
    {
        for (var k = 0; k < g.Nodes.Count; k++)
        {
            var n = g.Nodes[k];
            if (n.Id == id)
            {
                // If existing entry was a bare id reference (label == id and Rectangle)
                // and this occurrence carries a real shape/label, upgrade in place.
                if (bodyLabelSeen && n.Label == n.Id && n.Shape == NodeShape.Rectangle)
                {
                    g.Nodes[k] = n with { Label = label, Shape = shape };
                }
                return;
            }
        }
        g.Nodes.Add(new FlowchartNode(id, label, shape));
    }

    private static string? ParseIdent(string s, ref int i)
    {
        var start = i;
        while (i < s.Length && (char.IsLetterOrDigit(s[i]) || s[i] == '_' || s[i] == '-' || s[i] == '.'))
        {
            // Mermaid ids can include letters/digits/underscores; we also
            // allow '-' and '.' loosely to accept ids that appear in the wild.
            // But '--' / '-.-' / '-->' are edge tokens — stop before dashes
            // that begin an edge.
            if (s[i] == '-' && i + 1 < s.Length && (s[i + 1] == '-' || s[i + 1] == '.')) break;
            i++;
        }
        return i > start ? s.Substring(start, i - start) : null;
    }

    private static bool TryParseShape(string s, ref int i, out string label, out NodeShape shape)
    {
        label = "";
        shape = NodeShape.Rectangle;

        // Longer delimiters first: ([ , ((
        if (i + 1 < s.Length && s[i] == '(' && s[i + 1] == '[')
        {
            i += 2;
            if (!ReadUntil(s, ref i, "])", out label)) return false;
            i += 2;
            shape = NodeShape.Stadium;
            return true;
        }
        if (i + 1 < s.Length && s[i] == '(' && s[i + 1] == '(')
        {
            i += 2;
            if (!ReadUntil(s, ref i, "))", out label)) return false;
            i += 2;
            shape = NodeShape.Circle;
            return true;
        }
        if (s[i] == '[')
        {
            i++;
            if (!ReadUntil(s, ref i, "]", out label)) return false;
            i++;
            shape = NodeShape.Rectangle;
            return true;
        }
        if (s[i] == '(')
        {
            i++;
            if (!ReadUntil(s, ref i, ")", out label)) return false;
            i++;
            shape = NodeShape.Rounded;
            return true;
        }
        if (s[i] == '{')
        {
            i++;
            if (!ReadUntil(s, ref i, "}", out label)) return false;
            i++;
            shape = NodeShape.Rhombus;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Read the body of a shape. Supports optional <c>"..."</c> quoting so
    /// the body can contain the closing delimiter. Unquoted bodies stop at
    /// the first occurrence of <paramref name="terminator"/>.
    /// </summary>
    private static bool ReadUntil(string s, ref int i, string terminator, out string label)
    {
        SkipWs(s, ref i);
        // Quoted label
        if (i < s.Length && s[i] == '"')
        {
            i++;
            var qStart = i;
            while (i < s.Length && s[i] != '"') i++;
            if (i >= s.Length) { label = s.Substring(qStart); return false; }
            label = s.Substring(qStart, i - qStart);
            i++;  // past the close quote
            SkipWs(s, ref i);
            return i + terminator.Length <= s.Length && s.Substring(i, terminator.Length) == terminator;
        }

        var start = i;
        while (i + terminator.Length <= s.Length && s.Substring(i, terminator.Length) != terminator) i++;
        if (i + terminator.Length > s.Length) { label = s.Substring(start).TrimEnd(); return false; }
        label = s.Substring(start, i - start).Trim();
        return true;
    }

    private static bool TryParseEdge(string s, ref int i, out EdgeStyle style, out string? label)
    {
        style = EdgeStyle.SolidArrow;
        label = null;

        // Thick:  ==>    (two or more '=' then '>')
        if (i < s.Length && s[i] == '=')
        {
            var j = i;
            while (j < s.Length && s[j] == '=') j++;
            if (j < s.Length && s[j] == '>' && j - i >= 2)
            {
                i = j + 1;
                style = EdgeStyle.ThickArrow;
                TryConsumePipeLabel(s, ref i, out label);
                return true;
            }
            return false;
        }

        // Dotted: -.-> (one or more '-', then '.', then one or more '-', then '>')
        // Solid arrow: -->  (two or more '-' then '>')
        // Solid line: ---   (three or more '-')
        if (i < s.Length && s[i] == '-')
        {
            var j = i;
            // Consume leading dashes
            while (j < s.Length && s[j] == '-') j++;
            var leadingDashes = j - i;
            // Dotted middle
            if (j < s.Length && s[j] == '.' && leadingDashes >= 1)
            {
                j++;  // past '.'
                var trailStart = j;
                while (j < s.Length && s[j] == '-') j++;
                var trailingDashes = j - trailStart;
                if (trailingDashes < 1) return false;
                if (j < s.Length && s[j] == '>')
                {
                    i = j + 1;
                    style = EdgeStyle.DottedArrow;
                    TryConsumePipeLabel(s, ref i, out label);
                    return true;
                }
                // Dotted without arrow — not in our v1 vocabulary; treat as no-arrow solid
                i = j;
                style = EdgeStyle.SolidLine;
                TryConsumePipeLabel(s, ref i, out label);
                return true;
            }

            if (leadingDashes >= 2 && j < s.Length && s[j] == '>')
            {
                i = j + 1;
                style = EdgeStyle.SolidArrow;
                TryConsumePipeLabel(s, ref i, out label);
                return true;
            }
            if (leadingDashes >= 3)
            {
                i = j;
                style = EdgeStyle.SolidLine;
                TryConsumePipeLabel(s, ref i, out label);
                return true;
            }
            return false;
        }

        return false;
    }

    private static void TryConsumePipeLabel(string s, ref int i, out string? label)
    {
        label = null;
        SkipWs(s, ref i);
        if (i >= s.Length || s[i] != '|') return;
        i++;
        var start = i;
        while (i < s.Length && s[i] != '|') i++;
        if (i >= s.Length) { label = s.Substring(start).Trim(); return; }
        label = s.Substring(start, i - start).Trim();
        i++;  // past close pipe
    }

    private static void SkipWs(string s, ref int i)
    {
        while (i < s.Length && (s[i] == ' ' || s[i] == '\t')) i++;
    }
}
