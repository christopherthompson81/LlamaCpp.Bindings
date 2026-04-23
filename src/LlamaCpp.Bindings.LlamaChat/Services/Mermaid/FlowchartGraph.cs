using System.Collections.Generic;

namespace LlamaCpp.Bindings.LlamaChat.Services.Mermaid;

public enum FlowchartDirection
{
    TopDown,    // TD / TB
    BottomUp,   // BT
    LeftRight,  // LR
    RightLeft,  // RL
}

public enum NodeShape
{
    Rectangle,      // A[text]
    Rounded,        // A(text)
    Stadium,        // A([text])
    Circle,         // A((text))
    Rhombus,        // A{text}
}

public enum EdgeStyle
{
    SolidArrow,     // -->
    SolidLine,      // ---
    DottedArrow,    // -.->
    ThickArrow,     // ==>
}

public sealed record FlowchartNode(string Id, string Label, NodeShape Shape);

public sealed record FlowchartEdge(string FromId, string ToId, EdgeStyle Style, string? Label);

public sealed class FlowchartGraph
{
    public FlowchartDirection Direction { get; set; } = FlowchartDirection.TopDown;
    public List<FlowchartNode> Nodes { get; } = new();
    public List<FlowchartEdge> Edges { get; } = new();

    /// <summary>
    /// Lookup nodes by id; first declaration wins, later redeclarations of
    /// the same id without a fresh label/shape keep the original.
    /// </summary>
    public Dictionary<string, FlowchartNode> NodesById
    {
        get
        {
            var d = new Dictionary<string, FlowchartNode>(Nodes.Count);
            foreach (var n in Nodes)
                if (!d.ContainsKey(n.Id)) d[n.Id] = n;
            return d;
        }
    }
}
