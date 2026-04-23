using LlamaCpp.Bindings.LlamaChat.Services.Mermaid;

namespace LlamaCpp.Bindings.Tests;

public class FlowchartParserTests
{
    [Fact]
    public void Empty_Input_Produces_Empty_Graph()
    {
        var g = FlowchartParser.Parse("");
        Assert.Empty(g.Nodes);
        Assert.Empty(g.Edges);
        Assert.Equal(FlowchartDirection.TopDown, g.Direction);
    }

    [Theory]
    [InlineData("graph TD", FlowchartDirection.TopDown)]
    [InlineData("graph TB", FlowchartDirection.TopDown)]
    [InlineData("graph BT", FlowchartDirection.BottomUp)]
    [InlineData("graph LR", FlowchartDirection.LeftRight)]
    [InlineData("graph RL", FlowchartDirection.RightLeft)]
    [InlineData("flowchart LR", FlowchartDirection.LeftRight)]
    public void Header_Sets_Direction(string header, FlowchartDirection expected)
    {
        var g = FlowchartParser.Parse(header);
        Assert.Equal(expected, g.Direction);
    }

    [Fact]
    public void Missing_Header_Defaults_To_TopDown()
    {
        var g = FlowchartParser.Parse("A --> B");
        Assert.Equal(FlowchartDirection.TopDown, g.Direction);
        Assert.Equal(2, g.Nodes.Count);
        Assert.Single(g.Edges);
    }

    [Fact]
    public void Simple_Solid_Edge()
    {
        var g = FlowchartParser.Parse("graph TD\nA --> B");
        Assert.Equal(2, g.Nodes.Count);
        Assert.Equal("A", g.Nodes[0].Id);
        Assert.Equal("B", g.Nodes[1].Id);
        Assert.Single(g.Edges);
        Assert.Equal("A", g.Edges[0].FromId);
        Assert.Equal("B", g.Edges[0].ToId);
        Assert.Equal(EdgeStyle.SolidArrow, g.Edges[0].Style);
        Assert.Null(g.Edges[0].Label);
    }

    [Theory]
    [InlineData("A --> B", EdgeStyle.SolidArrow)]
    [InlineData("A --- B", EdgeStyle.SolidLine)]
    [InlineData("A -.-> B", EdgeStyle.DottedArrow)]
    [InlineData("A ==> B", EdgeStyle.ThickArrow)]
    [InlineData("A --------> B", EdgeStyle.SolidArrow)]
    public void Edge_Styles(string line, EdgeStyle expected)
    {
        var g = FlowchartParser.Parse(line);
        Assert.Single(g.Edges);
        Assert.Equal(expected, g.Edges[0].Style);
    }

    [Theory]
    [InlineData("A[Rect]", NodeShape.Rectangle, "Rect")]
    [InlineData("A(Rounded)", NodeShape.Rounded, "Rounded")]
    [InlineData("A([Stadium])", NodeShape.Stadium, "Stadium")]
    [InlineData("A((Circle))", NodeShape.Circle, "Circle")]
    [InlineData("A{Rhombus}", NodeShape.Rhombus, "Rhombus")]
    public void Node_Shapes(string line, NodeShape expectedShape, string expectedLabel)
    {
        var g = FlowchartParser.Parse(line);
        Assert.Single(g.Nodes);
        Assert.Equal(expectedShape, g.Nodes[0].Shape);
        Assert.Equal(expectedLabel, g.Nodes[0].Label);
    }

    [Fact]
    public void Node_Shape_Longer_Delimiters_Take_Priority()
    {
        // (( must not be parsed as ( with body "(...)"
        var g = FlowchartParser.Parse("A((inner))");
        Assert.Single(g.Nodes);
        Assert.Equal(NodeShape.Circle, g.Nodes[0].Shape);
        Assert.Equal("inner", g.Nodes[0].Label);
    }

    [Fact]
    public void Edge_Label_Pipe_Form()
    {
        var g = FlowchartParser.Parse("A -->|yes| B");
        Assert.Single(g.Edges);
        Assert.Equal("yes", g.Edges[0].Label);
    }

    [Fact]
    public void Edge_Chain_Produces_Multiple_Edges()
    {
        var g = FlowchartParser.Parse("A --> B --> C");
        Assert.Equal(3, g.Nodes.Count);
        Assert.Equal(2, g.Edges.Count);
        Assert.Equal(("A", "B"), (g.Edges[0].FromId, g.Edges[0].ToId));
        Assert.Equal(("B", "C"), (g.Edges[1].FromId, g.Edges[1].ToId));
    }

    [Fact]
    public void Inline_Node_Definitions_On_Edge()
    {
        var g = FlowchartParser.Parse("A[Start] --> B{Choice}");
        Assert.Equal(2, g.Nodes.Count);
        Assert.Equal(NodeShape.Rectangle, g.Nodes[0].Shape);
        Assert.Equal("Start", g.Nodes[0].Label);
        Assert.Equal(NodeShape.Rhombus, g.Nodes[1].Shape);
        Assert.Equal("Choice", g.Nodes[1].Label);
    }

    [Fact]
    public void Node_First_Labelled_Occurrence_Wins()
    {
        var g = FlowchartParser.Parse("A --> B\nA[Start] --> C");
        var a = g.Nodes.Find(n => n.Id == "A")!;
        Assert.Equal("Start", a.Label);
        Assert.Equal(NodeShape.Rectangle, a.Shape);
    }

    [Fact]
    public void Quoted_Node_Label_Supports_Special_Chars()
    {
        var g = FlowchartParser.Parse("A[\"Hello, [world]!\"] --> B");
        Assert.Equal("Hello, [world]!", g.Nodes[0].Label);
    }

    [Fact]
    public void Comments_Are_Ignored()
    {
        var g = FlowchartParser.Parse("graph TD\n%% this is a comment\nA --> B");
        Assert.Single(g.Edges);
    }

    [Fact]
    public void Unsupported_Lines_Are_Swallowed()
    {
        var source = """
            graph TD
            A --> B
            classDef green fill:#9f6
            class A green
            click A callback "tooltip"
            linkStyle 0 stroke:#f00
            C --> D
            """;
        var g = FlowchartParser.Parse(source);
        Assert.Equal(4, g.Nodes.Count);  // A, B, C, D
        Assert.Equal(2, g.Edges.Count);
    }

    [Fact]
    public void Realistic_Small_Flowchart()
    {
        var source = """
            flowchart TD
                Start([Start]) --> Check{Ready?}
                Check -->|yes| Run[Run the thing]
                Check -->|no| Wait(Wait)
                Run --> Done([Done])
                Wait --> Check
            """;
        var g = FlowchartParser.Parse(source);
        Assert.Equal(FlowchartDirection.TopDown, g.Direction);
        Assert.Equal(5, g.Nodes.Count);
        Assert.Equal(5, g.Edges.Count);
        Assert.Contains(g.Nodes, n => n.Id == "Start" && n.Shape == NodeShape.Stadium);
        Assert.Contains(g.Nodes, n => n.Id == "Check" && n.Shape == NodeShape.Rhombus);
        Assert.Contains(g.Edges, e => e.FromId == "Check" && e.ToId == "Run" && e.Label == "yes");
        Assert.Contains(g.Edges, e => e.FromId == "Wait" && e.ToId == "Check");  // back-edge (cycle)
    }
}
