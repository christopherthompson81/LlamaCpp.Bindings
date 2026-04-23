using LlamaCpp.Bindings.LlamaChat.Services.Mermaid;

namespace LlamaCpp.Bindings.Tests;

public class FlowchartLayoutTests
{
    // Fixed label measurer: 7px per char wide, 14px tall. Mirrors the
    // approximate metrics of a 12pt system font — precise values don't
    // matter here, we're testing structure not pixels.
    private static readonly FlowchartLayout.Measurer Measure =
        s => (s.Length * 7.0, 14.0);

    [Fact]
    public void Empty_Graph_Produces_Empty_Layout()
    {
        var g = new FlowchartGraph();
        var lo = FlowchartLayout.Compute(g, Measure);
        Assert.Empty(lo.Nodes);
        Assert.Empty(lo.Edges);
    }

    [Fact]
    public void Linear_Chain_Layers_Top_Down()
    {
        var g = FlowchartParser.Parse("graph TD\nA --> B --> C");
        var lo = FlowchartLayout.Compute(g, Measure);
        Assert.Equal(3, lo.Nodes.Count);
        var a = lo.Nodes.Single(n => n.Node.Id == "A");
        var b = lo.Nodes.Single(n => n.Node.Id == "B");
        var c = lo.Nodes.Single(n => n.Node.Id == "C");
        // TD means increasing Y per layer.
        Assert.True(a.Y < b.Y, "A should sit above B");
        Assert.True(b.Y < c.Y, "B should sit above C");
    }

    [Fact]
    public void Linear_Chain_Layers_Left_Right()
    {
        var g = FlowchartParser.Parse("graph LR\nA --> B --> C");
        var lo = FlowchartLayout.Compute(g, Measure);
        var a = lo.Nodes.Single(n => n.Node.Id == "A");
        var b = lo.Nodes.Single(n => n.Node.Id == "B");
        var c = lo.Nodes.Single(n => n.Node.Id == "C");
        Assert.True(a.X < b.X);
        Assert.True(b.X < c.X);
    }

    [Fact]
    public void Cycle_Still_Lays_Out()
    {
        // A → B → A  — one back-edge. Both nodes should still appear.
        var g = FlowchartParser.Parse("A --> B\nB --> A");
        var lo = FlowchartLayout.Compute(g, Measure);
        Assert.Equal(2, lo.Nodes.Count);
        Assert.Equal(2, lo.Edges.Count);
    }

    [Fact]
    public void Fan_Out_Siblings_Share_A_Layer()
    {
        var g = FlowchartParser.Parse("graph TD\nA --> B\nA --> C\nA --> D");
        var lo = FlowchartLayout.Compute(g, Measure);
        var b = lo.Nodes.Single(n => n.Node.Id == "B");
        var c = lo.Nodes.Single(n => n.Node.Id == "C");
        var d = lo.Nodes.Single(n => n.Node.Id == "D");
        // B, C, D all on same Y layer (within ~5px tolerance for padding differences).
        Assert.InRange(Math.Abs(b.Y - c.Y), 0, 5);
        Assert.InRange(Math.Abs(c.Y - d.Y), 0, 5);
    }

    [Fact]
    public void Circle_Shape_Produces_Square_Bounding_Box()
    {
        var g = FlowchartParser.Parse("A((label))");
        var lo = FlowchartLayout.Compute(g, Measure);
        var a = lo.Nodes.Single();
        Assert.Equal(a.Width, a.Height, precision: 3);
    }

    [Fact]
    public void Edge_Endpoints_Sit_On_Shape_Boundaries()
    {
        var g = FlowchartParser.Parse("graph TD\nA --> B");
        var lo = FlowchartLayout.Compute(g, Measure);
        var a = lo.Nodes.Single(n => n.Node.Id == "A");
        var b = lo.Nodes.Single(n => n.Node.Id == "B");
        var e = lo.Edges.Single();
        var (srcX, srcY) = e.Points[0];
        var (dstX, dstY) = e.Points[1];
        // For TD layout the edge leaves A's bottom and enters B's top,
        // roughly through the middle X of each node.
        Assert.InRange(srcY, a.Y + a.Height - 0.01, a.Y + a.Height + 0.01);
        Assert.InRange(dstY, b.Y - 0.01, b.Y + 0.01);
        Assert.InRange(srcX, a.X, a.X + a.Width);
        Assert.InRange(dstX, b.X, b.X + b.Width);
    }

    [Fact]
    public void Bounding_Box_Is_Positive_And_Encloses_Nodes()
    {
        var g = FlowchartParser.Parse("graph TD\nA --> B\nA --> C");
        var lo = FlowchartLayout.Compute(g, Measure);
        Assert.True(lo.Width > 0);
        Assert.True(lo.Height > 0);
        foreach (var n in lo.Nodes)
        {
            Assert.True(n.X >= 0);
            Assert.True(n.Y >= 0);
            Assert.True(n.X + n.Width <= lo.Width + 0.01);
            Assert.True(n.Y + n.Height <= lo.Height + 0.01);
        }
    }
}
