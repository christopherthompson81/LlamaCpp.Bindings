using System;
using System.Collections.Generic;
using System.Linq;

namespace LlamaCpp.Bindings.LlamaChat.Services.Mermaid;

/// <summary>
/// Compute positions for a <see cref="FlowchartGraph"/> via a small Sugiyama
/// layered layout:
/// <list type="number">
///   <item>Cycle removal by DFS back-edge marking (those edges are rendered
///     but treated as forward for layer computation).</item>
///   <item>Layer assignment by longest-path from sources (sinks-if-no-sources
///     fallback).</item>
///   <item>Crossing reduction by barycenter sweeps, alternating directions.</item>
///   <item>Coordinate assignment evenly-spaced within each layer, centred
///     across the widest layer.</item>
/// </list>
/// <para>
/// Deliberately Avalonia-free so it can be unit-tested in the bindings test
/// project. Label sizing is injected via <see cref="Measurer"/>.
/// </para>
/// </summary>
public static class FlowchartLayout
{
    public delegate (double Width, double Height) Measurer(string text);

    /// <summary>Gap between neighbouring layers (perpendicular to layer axis).</summary>
    public const double LayerGap = 60;
    /// <summary>Gap between siblings within a layer.</summary>
    public const double SiblingGap = 28;
    /// <summary>Minimum padding around a label inside a shape.</summary>
    public const double PaddingX = 14;
    public const double PaddingY = 10;

    public static LaidOutGraph Compute(FlowchartGraph g, Measurer measure)
    {
        var layout = new LaidOutGraph { Direction = g.Direction };
        if (g.Nodes.Count == 0) return layout;

        var nodeById = g.NodesById;
        var ids = g.Nodes.Select(n => n.Id).ToArray();
        var idIndex = new Dictionary<string, int>(ids.Length);
        for (var i = 0; i < ids.Length; i++) idIndex[ids[i]] = i;

        // Adjacency (forward + reverse). Edges whose target id is missing
        // from the declared node set are silently skipped — the parser should
        // have added them, but guard anyway.
        var outs = new List<int>[ids.Length];
        var ins = new List<int>[ids.Length];
        for (var i = 0; i < ids.Length; i++) { outs[i] = new(); ins[i] = new(); }

        var edgesForLayout = new List<(int from, int to, FlowchartEdge edge, bool reversed)>();
        foreach (var e in g.Edges)
        {
            if (!idIndex.TryGetValue(e.FromId, out var f)) continue;
            if (!idIndex.TryGetValue(e.ToId, out var t)) continue;
            outs[f].Add(t);
            ins[t].Add(f);
            edgesForLayout.Add((f, t, e, false));
        }

        // 1. Cycle removal: DFS from every node, mark back-edges. Reverse them
        //    in a working adjacency used only for layering; keep the original
        //    direction for drawing.
        var backEdges = FindBackEdges(ids.Length, outs);
        var layeringOuts = new List<int>[ids.Length];
        var layeringIns = new List<int>[ids.Length];
        for (var i = 0; i < ids.Length; i++) { layeringOuts[i] = new(); layeringIns[i] = new(); }
        foreach (var (f, t, _, _) in edgesForLayout)
        {
            if (backEdges.Contains((f, t)))
            {
                layeringOuts[t].Add(f);
                layeringIns[f].Add(t);
            }
            else
            {
                layeringOuts[f].Add(t);
                layeringIns[t].Add(f);
            }
        }

        // 2. Layer assignment — longest path from sources.
        var layer = AssignLayers(ids.Length, layeringOuts, layeringIns);
        var maxLayer = layer.Length == 0 ? 0 : layer.Max();

        // 3. Group nodes by layer, keep initial declaration order.
        var byLayer = new List<List<int>>();
        for (var L = 0; L <= maxLayer; L++) byLayer.Add(new());
        for (var i = 0; i < ids.Length; i++) byLayer[layer[i]].Add(i);

        // 4. Crossing reduction — barycenter sweeps.
        ReduceCrossings(byLayer, layeringOuts, layeringIns);

        // 5. Measure node sizes.
        var sizes = new (double W, double H)[ids.Length];
        for (var i = 0; i < ids.Length; i++)
        {
            var node = nodeById[ids[i]];
            var (tw, th) = measure(node.Label);
            sizes[i] = NodeBox(tw, th, node.Shape);
        }

        // 6. Assign unitless coordinates (layerAxis, crossAxis) per direction.
        var nodePos = new (double layerAxis, double crossAxis)[ids.Length];
        var layerExtent = new double[byLayer.Count];  // max size along the layer axis for that layer
        var crossTotal = new double[byLayer.Count];  // total size along cross axis

        for (var L = 0; L < byLayer.Count; L++)
        {
            var siblings = byLayer[L];
            double crossSum = 0;
            for (var s = 0; s < siblings.Count; s++)
            {
                var idx = siblings[s];
                var (w, h) = sizes[idx];
                var crossSize = CrossAxisSize(w, h, g.Direction);
                var layerSize = LayerAxisSize(w, h, g.Direction);
                if (layerSize > layerExtent[L]) layerExtent[L] = layerSize;
                if (s > 0) crossSum += SiblingGap;
                nodePos[idx] = (0, crossSum + crossSize / 2.0);
                crossSum += crossSize;
            }
            crossTotal[L] = crossSum;
        }

        // 7. Compute layer origin offsets along the layer axis.
        var layerOrigin = new double[byLayer.Count];
        double cursor = 0;
        for (var L = 0; L < byLayer.Count; L++)
        {
            layerOrigin[L] = cursor + layerExtent[L] / 2.0;
            cursor += layerExtent[L] + LayerGap;
        }

        // 8. Centre each layer's cross-axis extent across the widest layer.
        var maxCross = crossTotal.DefaultIfEmpty(0).Max();
        for (var L = 0; L < byLayer.Count; L++)
        {
            var offset = (maxCross - crossTotal[L]) / 2.0;
            foreach (var idx in byLayer[L])
            {
                var (la, ca) = nodePos[idx];
                nodePos[idx] = (layerOrigin[L], ca + offset);
            }
        }

        // 9. Project (layerAxis, crossAxis) → (X, Y) based on direction, and
        //    invert if needed so BT reads bottom-up / RL right-to-left.
        var totalLayer = cursor - LayerGap;  // omit trailing gap
        foreach (var (idx, pos) in nodePos.Select((p, i) => (i, p)))
        {
            var (w, h) = sizes[idx];
            double x, y;
            switch (g.Direction)
            {
                case FlowchartDirection.TopDown:
                    x = pos.crossAxis - w / 2.0;
                    y = pos.layerAxis - h / 2.0;
                    break;
                case FlowchartDirection.BottomUp:
                    x = pos.crossAxis - w / 2.0;
                    y = (totalLayer - pos.layerAxis) - h / 2.0;
                    break;
                case FlowchartDirection.LeftRight:
                    x = pos.layerAxis - w / 2.0;
                    y = pos.crossAxis - h / 2.0;
                    break;
                case FlowchartDirection.RightLeft:
                    x = (totalLayer - pos.layerAxis) - w / 2.0;
                    y = pos.crossAxis - h / 2.0;
                    break;
                default:
                    x = pos.crossAxis - w / 2.0;
                    y = pos.layerAxis - h / 2.0;
                    break;
            }
            var node = nodeById[ids[idx]];
            layout.Nodes.Add(new LaidOutNode
            {
                Node = node,
                X = x,
                Y = y,
                Width = w,
                Height = h,
            });
        }

        // 10. Compute edge paths. v1: straight line from source centre to
        //     target centre, clipped to each shape's boundary so the
        //     arrowhead sits on the edge rather than inside the box.
        foreach (var (f, t, e, _) in edgesForLayout)
        {
            var src = layout.Nodes[f];
            var dst = layout.Nodes[t];
            var srcCenter = (X: src.X + src.Width / 2.0, Y: src.Y + src.Height / 2.0);
            var dstCenter = (X: dst.X + dst.Width / 2.0, Y: dst.Y + dst.Height / 2.0);
            var srcExit = ClipToShape(srcCenter, dstCenter, src);
            var dstEntry = ClipToShape(dstCenter, srcCenter, dst);
            layout.Edges.Add(new LaidOutEdge
            {
                Edge = e,
                Points = new[] { srcExit, dstEntry },
            });
        }

        // 11. Bounding box
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var n in layout.Nodes)
        {
            if (n.X < minX) minX = n.X;
            if (n.Y < minY) minY = n.Y;
            if (n.X + n.Width > maxX) maxX = n.X + n.Width;
            if (n.Y + n.Height > maxY) maxY = n.Y + n.Height;
        }
        if (minX == double.MaxValue) { minX = 0; minY = 0; maxX = 0; maxY = 0; }

        // Shift so the top-left corner sits at (0, 0) with a small margin.
        const double margin = 8;
        var dx = -minX + margin;
        var dy = -minY + margin;
        foreach (var n in layout.Nodes)
        {
            n.X += dx;
            n.Y += dy;
        }
        foreach (var e in layout.Edges)
        {
            for (var i = 0; i < e.Points.Length; i++)
                e.Points[i] = (e.Points[i].X + dx, e.Points[i].Y + dy);
        }
        layout.Width = (maxX - minX) + margin * 2;
        layout.Height = (maxY - minY) + margin * 2;

        return layout;
    }

    private static (double W, double H) NodeBox(double textW, double textH, NodeShape shape)
    {
        var w = textW + PaddingX * 2;
        var h = textH + PaddingY * 2;
        switch (shape)
        {
            case NodeShape.Circle:
                var d = Math.Max(w, h);
                return (d, d);
            case NodeShape.Rhombus:
                // Rhombus fits the label with extra diagonal slack so it
                // doesn't clip at the corners.
                return (w * 1.4, h * 1.4);
            case NodeShape.Stadium:
                return (w + h * 0.5, h);
            default:
                return (w, h);
        }
    }

    private static double LayerAxisSize(double w, double h, FlowchartDirection d) =>
        d is FlowchartDirection.LeftRight or FlowchartDirection.RightLeft ? w : h;

    private static double CrossAxisSize(double w, double h, FlowchartDirection d) =>
        d is FlowchartDirection.LeftRight or FlowchartDirection.RightLeft ? h : w;

    // ---- cycle removal ----

    private static HashSet<(int, int)> FindBackEdges(int n, List<int>[] outs)
    {
        var state = new byte[n];  // 0=white, 1=gray, 2=black
        var back = new HashSet<(int, int)>();
        for (var start = 0; start < n; start++)
        {
            if (state[start] != 0) continue;
            var stack = new Stack<(int node, int iter)>();
            stack.Push((start, 0));
            state[start] = 1;
            while (stack.Count > 0)
            {
                var (u, k) = stack.Pop();
                if (k < outs[u].Count)
                {
                    stack.Push((u, k + 1));
                    var v = outs[u][k];
                    if (state[v] == 1) back.Add((u, v));
                    else if (state[v] == 0)
                    {
                        state[v] = 1;
                        stack.Push((v, 0));
                    }
                }
                else
                {
                    state[u] = 2;
                }
            }
        }
        return back;
    }

    // ---- layer assignment ----

    private static int[] AssignLayers(int n, List<int>[] outs, List<int>[] ins)
    {
        // Longest-path topological: sources at layer 0.
        var inDeg = new int[n];
        for (var i = 0; i < n; i++) inDeg[i] = ins[i].Count;
        var ready = new Queue<int>();
        for (var i = 0; i < n; i++) if (inDeg[i] == 0) ready.Enqueue(i);

        var layer = new int[n];
        var processed = 0;
        while (ready.Count > 0)
        {
            var u = ready.Dequeue();
            processed++;
            foreach (var v in outs[u])
            {
                if (layer[u] + 1 > layer[v]) layer[v] = layer[u] + 1;
                if (--inDeg[v] == 0) ready.Enqueue(v);
            }
        }
        // If cycles remain (shouldn't after back-edge reversal), pile unprocessed
        // nodes onto the last layer so they at least render.
        if (processed < n)
        {
            var maxL = 0;
            for (var i = 0; i < n; i++) if (layer[i] > maxL) maxL = layer[i];
            for (var i = 0; i < n; i++)
                if (inDeg[i] != 0) layer[i] = maxL + 1;
        }
        return layer;
    }

    // ---- crossing reduction ----

    private static void ReduceCrossings(List<List<int>> byLayer, List<int>[] outs, List<int>[] ins)
    {
        const int passes = 12;
        for (var p = 0; p < passes; p++)
        {
            if (p % 2 == 0)
            {
                // Down sweep: reorder layer L by barycenter of predecessors in L-1
                for (var L = 1; L < byLayer.Count; L++) SortByBarycenter(byLayer[L], byLayer[L - 1], ins);
            }
            else
            {
                // Up sweep: reorder layer L by barycenter of successors in L+1
                for (var L = byLayer.Count - 2; L >= 0; L--) SortByBarycenter(byLayer[L], byLayer[L + 1], outs);
            }
        }
    }

    private static void SortByBarycenter(List<int> target, List<int> reference, List<int>[] neighbours)
    {
        var refIdx = new Dictionary<int, int>(reference.Count);
        for (var i = 0; i < reference.Count; i++) refIdx[reference[i]] = i;

        var bary = new double[target.Count];
        for (var i = 0; i < target.Count; i++)
        {
            var u = target[i];
            var ns = neighbours[u];
            double sum = 0;
            var count = 0;
            foreach (var v in ns)
            {
                if (refIdx.TryGetValue(v, out var idx)) { sum += idx; count++; }
            }
            bary[i] = count == 0 ? i : sum / count;
        }

        var indexed = Enumerable.Range(0, target.Count).ToArray();
        Array.Sort(indexed, (a, b) => bary[a].CompareTo(bary[b]));
        var reordered = new List<int>(target.Count);
        foreach (var idx in indexed) reordered.Add(target[idx]);
        target.Clear();
        target.AddRange(reordered);
    }

    // ---- clip line from a box centre to a point on its shape boundary ----

    private static (double X, double Y) ClipToShape((double X, double Y) from, (double X, double Y) to, LaidOutNode n)
    {
        var dx = to.X - from.X;
        var dy = to.Y - from.Y;
        if (dx == 0 && dy == 0) return from;

        // All shapes fit inside their (Width, Height) bounding box. For v1 we
        // clip to the box; circles get a true circle clip for visual
        // correctness, rhombus gets a rhombus clip.
        var w = n.Width / 2.0;
        var h = n.Height / 2.0;
        switch (n.Node.Shape)
        {
            case NodeShape.Circle:
            {
                var r = Math.Min(w, h);
                var len = Math.Sqrt(dx * dx + dy * dy);
                return (from.X + dx / len * r, from.Y + dy / len * r);
            }
            case NodeShape.Rhombus:
            {
                // |x/w| + |y/h| = 1 → t such that from + t*(dx,dy) hits that.
                var t = 1.0 / (Math.Abs(dx) / w + Math.Abs(dy) / h);
                return (from.X + dx * t, from.Y + dy * t);
            }
            default:
            {
                var tx = dx == 0 ? double.MaxValue : w / Math.Abs(dx);
                var ty = dy == 0 ? double.MaxValue : h / Math.Abs(dy);
                var t = Math.Min(tx, ty);
                return (from.X + dx * t, from.Y + dy * t);
            }
        }
    }
}

public sealed class LaidOutNode
{
    public required FlowchartNode Node { get; init; }
    public double X;
    public double Y;
    public double Width;
    public double Height;
}

public sealed class LaidOutEdge
{
    public required FlowchartEdge Edge { get; init; }
    public required (double X, double Y)[] Points;
}

public sealed class LaidOutGraph
{
    public FlowchartDirection Direction { get; set; }
    public List<LaidOutNode> Nodes { get; } = new();
    public List<LaidOutEdge> Edges { get; } = new();
    public double Width;
    public double Height;
}
