using System;
using Avalonia;
using Avalonia.Collections;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Layout;
using Avalonia.Markup.Xaml.MarkupExtensions;
using Avalonia.Media;

namespace LlamaCpp.Bindings.LlamaChat.Services.Mermaid;

/// <summary>
/// Walk a <see cref="LaidOutGraph"/> and emit an Avalonia <see cref="Canvas"/>
/// with each node as a shape + centred label, and each edge as a line with
/// optional arrowhead + midpoint label. Styling pulls from the theme's
/// <c>Foreground</c>, <c>MutedForeground</c>, <c>Card</c>, and <c>Border</c>
/// brushes so flowcharts track the current theme.
/// </summary>
public static class FlowchartRenderer
{
    private const double FontSize = 12;
    private const double EdgeStrokeThickness = 1.5;
    private const double ThickStrokeThickness = 3.0;
    private const double ArrowheadLength = 10;
    private const double ArrowheadHalfWidth = 5;

    public static Control Render(string source)
    {
        var graph = FlowchartParser.Parse(source);
        if (graph.Nodes.Count == 0)
        {
            return new TextBlock
            {
                Text = "(empty flowchart)",
                Opacity = 0.6,
                Margin = new Thickness(8),
            };
        }

        var typeface = new Typeface(FontFamily.Default);
        FlowchartLayout.Measurer measure = s =>
        {
            var ft = new FormattedText(
                s, System.Globalization.CultureInfo.CurrentCulture,
                FlowDirection.LeftToRight, typeface, FontSize, Brushes.Black);
            return (ft.Width, ft.Height);
        };

        var laid = FlowchartLayout.Compute(graph, measure);

        var canvas = new Canvas
        {
            Width = laid.Width,
            Height = laid.Height,
        };

        // Edges first so nodes paint over endpoints.
        foreach (var e in laid.Edges) AddEdge(canvas, e, laid);
        foreach (var n in laid.Nodes) AddNode(canvas, n);

        var container = new Border
        {
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(12),
            Margin = new Thickness(0, 6, 0, 6),
            Child = new ScrollViewer
            {
                HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
                VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Disabled,
                Content = canvas,
            },
        };
        container[!Border.BackgroundProperty] = new DynamicResourceExtension("CodeBackground");
        return container;
    }

    private static void AddNode(Canvas canvas, LaidOutNode n)
    {
        Shape shape = n.Node.Shape switch
        {
            NodeShape.Rectangle => new Rectangle { Width = n.Width, Height = n.Height },
            NodeShape.Rounded => new Rectangle { Width = n.Width, Height = n.Height, RadiusX = 10, RadiusY = 10 },
            NodeShape.Stadium => new Rectangle { Width = n.Width, Height = n.Height, RadiusX = n.Height / 2.0, RadiusY = n.Height / 2.0 },
            NodeShape.Circle => new Ellipse { Width = n.Width, Height = n.Height },
            NodeShape.Rhombus => RhombusPath(n.Width, n.Height),
            _ => new Rectangle { Width = n.Width, Height = n.Height },
        };
        shape.StrokeThickness = 1.3;
        shape[!Shape.StrokeProperty] = new DynamicResourceExtension("Foreground");
        shape[!Shape.FillProperty] = new DynamicResourceExtension("Card");

        Canvas.SetLeft(shape, n.X);
        Canvas.SetTop(shape, n.Y);
        canvas.Children.Add(shape);

        var label = new TextBlock
        {
            Text = n.Node.Label,
            FontSize = FontSize,
            Width = n.Width,
            Height = n.Height,
            TextAlignment = TextAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            TextWrapping = TextWrapping.NoWrap,
            Padding = new Thickness(0, (n.Height - FontSize * 1.2) / 2.0, 0, 0),
        };
        label[!TextBlock.ForegroundProperty] = new DynamicResourceExtension("Foreground");
        Canvas.SetLeft(label, n.X);
        Canvas.SetTop(label, n.Y);
        canvas.Children.Add(label);
    }

    private static Path RhombusPath(double w, double h)
    {
        var figure = new PathFigure
        {
            StartPoint = new Point(w / 2.0, 0),
            IsClosed = true,
        };
        figure.Segments!.Add(new LineSegment { Point = new Point(w, h / 2.0) });
        figure.Segments.Add(new LineSegment { Point = new Point(w / 2.0, h) });
        figure.Segments.Add(new LineSegment { Point = new Point(0, h / 2.0) });
        var geom = new PathGeometry();
        geom.Figures!.Add(figure);
        return new Path { Data = geom };
    }

    private static void AddEdge(Canvas canvas, LaidOutEdge e, LaidOutGraph laid)
    {
        if (e.Points.Length < 2) return;

        var (sx, sy) = e.Points[0];
        var (tx, ty) = e.Points[^1];

        // Cubic Bezier control points, biased along the layout's layer axis so
        // edges leave the source and enter the target tangent to that axis.
        // If the resulting curve would slice through an unrelated node's
        // bounding box, bulge the control points outward until the curve
        // clears the obstacle (or we run out of attempts).
        var (c1x, c1y, c2x, c2y) = BezierControls(sx, sy, tx, ty, laid, e.Source, e.Target);

        var path = new Path
        {
            StrokeThickness = e.Edge.Style == EdgeStyle.ThickArrow ? ThickStrokeThickness : EdgeStrokeThickness,
            Data = BuildBezierGeometry(sx, sy, c1x, c1y, c2x, c2y, tx, ty),
        };
        path[!Shape.StrokeProperty] = new DynamicResourceExtension("Foreground");
        if (e.Edge.Style == EdgeStyle.DottedArrow)
        {
            path.StrokeDashArray = new AvaloniaList<double> { 2, 3 };
        }
        canvas.Children.Add(path);

        // Arrowhead aligned to the Bezier's tangent at t=1 (direction from C2 to end).
        if (e.Edge.Style != EdgeStyle.SolidLine)
        {
            canvas.Children.Add(Arrowhead(c2x, c2y, tx, ty));
        }

        // Edge label at the midpoint.
        if (!string.IsNullOrWhiteSpace(e.Edge.Label))
        {
            var label = new Border
            {
                Padding = new Thickness(4, 1),
                CornerRadius = new CornerRadius(3),
                Child = new TextBlock
                {
                    Text = e.Edge.Label,
                    FontSize = FontSize - 2,
                    TextWrapping = TextWrapping.NoWrap,
                    [!TextBlock.ForegroundProperty] = new DynamicResourceExtension("Foreground"),
                },
            };
            label[!Border.BackgroundProperty] = new DynamicResourceExtension("CodeBackground");
            label.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            // Bezier midpoint at t = 0.5:
            //   P(0.5) = 0.125*(P0 + P3) + 0.375*(C1 + C2)
            var bmx = 0.125 * (sx + tx) + 0.375 * (c1x + c2x);
            var bmy = 0.125 * (sy + ty) + 0.375 * (c1y + c2y);
            Canvas.SetLeft(label, bmx - label.DesiredSize.Width / 2.0);
            Canvas.SetTop(label, bmy - label.DesiredSize.Height / 2.0);
            canvas.Children.Add(label);
        }
    }

    /// <summary>
    /// Compute cubic Bezier control points so the curve leaves the source
    /// and enters the target tangent to the layout's layer axis.
    ///
    /// Obstacle-aware: the tangent magnitude stays fixed at the aesthetic
    /// baseline (40% of the axis span). What changes on retry is a
    /// *perpendicular* bulge applied to both control points — shifted in the
    /// cross-axis direction away from any obstacles the baseline curve hits.
    /// Stretching the tangent instead would just extend the curve along the
    /// layer axis, which rarely helps when the obstacle sits beside the
    /// layer-axis midline.
    /// </summary>
    private static (double C1x, double C1y, double C2x, double C2y) BezierControls(
        double sx, double sy, double tx, double ty,
        LaidOutGraph laid, LaidOutNode? srcNode, LaidOutNode? dstNode)
    {
        var (bc1x, bc1y, bc2x, bc2y) = TangentControls(sx, sy, tx, ty, laid.Direction);

        var violations = CountObstacleHits(sx, sy, bc1x, bc1y, bc2x, bc2y, tx, ty,
                                           laid.Nodes, srcNode, dstNode, laid.Direction,
                                           out var avgHitCross);
        if (violations == 0) return (bc1x, bc1y, bc2x, bc2y);

        var sourceCross = CrossCoord(sx, sy, laid.Direction);
        var sign = avgHitCross >= sourceCross ? -1.0 : 1.0;

        // Try progressively stronger perpendicular shifts. The step size is
        // tied to LayerGap so bulges scale with the graph density rather than
        // the tangent magnitude.
        double[] perpMultipliers = { 0.6, 1.0, 1.6, 2.4 };
        (double c1x, double c1y, double c2x, double c2y) best = (bc1x, bc1y, bc2x, bc2y);
        var bestHits = violations;
        foreach (var pm in perpMultipliers)
        {
            var (dx, dy) = PerpOffset(laid.Direction, pm);
            var c1x = bc1x + dx * sign;
            var c1y = bc1y + dy * sign;
            var c2x = bc2x + dx * sign;
            var c2y = bc2y + dy * sign;
            var hits = CountObstacleHits(sx, sy, c1x, c1y, c2x, c2y, tx, ty,
                                         laid.Nodes, srcNode, dstNode, laid.Direction, out _);
            if (hits == 0) return (c1x, c1y, c2x, c2y);
            if (hits < bestHits)
            {
                bestHits = hits;
                best = (c1x, c1y, c2x, c2y);
            }
        }
        return best;
    }

    private static (double C1x, double C1y, double C2x, double C2y) TangentControls(
        double sx, double sy, double tx, double ty, FlowchartDirection dir)
    {
        const double BaselineMag = 0.4;
        var verticalAxis = dir is FlowchartDirection.TopDown or FlowchartDirection.BottomUp;
        if (verticalAxis)
        {
            var dy = ty - sy;
            var sign = dy >= 0 ? 1.0 : -1.0;
            var offset = Math.Max(Math.Abs(dy) * BaselineMag, FlowchartLayout.LayerGap * 0.35);
            return (sx, sy + sign * offset, tx, ty - sign * offset);
        }
        else
        {
            var dx = tx - sx;
            var sign = dx >= 0 ? 1.0 : -1.0;
            var offset = Math.Max(Math.Abs(dx) * BaselineMag, FlowchartLayout.LayerGap * 0.35);
            return (sx + sign * offset, sy, tx - sign * offset, ty);
        }
    }

    private static (double dx, double dy) PerpOffset(FlowchartDirection dir, double mag) =>
        dir is FlowchartDirection.TopDown or FlowchartDirection.BottomUp
            ? (FlowchartLayout.LayerGap * mag, 0.0)
            : (0.0, FlowchartLayout.LayerGap * mag);

    private static double CrossCoord(double x, double y, FlowchartDirection dir) =>
        dir is FlowchartDirection.TopDown or FlowchartDirection.BottomUp ? x : y;

    /// <summary>
    /// Count how many curve samples fall inside a non-endpoint node's bbox,
    /// and report the average cross-axis coordinate of those hits so the
    /// caller can bulge the curve toward the clearer side.
    /// </summary>
    private static int CountObstacleHits(
        double sx, double sy, double c1x, double c1y, double c2x, double c2y, double tx, double ty,
        System.Collections.Generic.IReadOnlyList<LaidOutNode> nodes,
        LaidOutNode? srcNode, LaidOutNode? dstNode, FlowchartDirection dir,
        out double avgHitCross)
    {
        const int Samples = 16;
        const double Inflate = 4;
        var hits = 0;
        double crossSum = 0;
        var vertical = dir is FlowchartDirection.TopDown or FlowchartDirection.BottomUp;
        for (var k = 1; k < Samples; k++)
        {
            var t = k / (double)Samples;
            var u = 1 - t;
            var bx = u * u * u * sx + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * tx;
            var by = u * u * u * sy + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * ty;
            foreach (var n in nodes)
            {
                if (ReferenceEquals(n, srcNode) || ReferenceEquals(n, dstNode)) continue;
                if (bx > n.X - Inflate && bx < n.X + n.Width + Inflate
                    && by > n.Y - Inflate && by < n.Y + n.Height + Inflate)
                {
                    hits++;
                    // cross-axis coord of the obstacle's centre
                    crossSum += vertical ? (n.X + n.Width / 2.0) : (n.Y + n.Height / 2.0);
                    break;
                }
            }
        }
        avgHitCross = hits == 0 ? 0 : crossSum / hits;
        return hits;
    }

    private static PathGeometry BuildBezierGeometry(
        double sx, double sy, double c1x, double c1y, double c2x, double c2y, double tx, double ty)
    {
        var figure = new PathFigure
        {
            StartPoint = new Point(sx, sy),
            IsClosed = false,
        };
        figure.Segments!.Add(new BezierSegment
        {
            Point1 = new Point(c1x, c1y),
            Point2 = new Point(c2x, c2y),
            Point3 = new Point(tx, ty),
        });
        var geom = new PathGeometry();
        geom.Figures!.Add(figure);
        return geom;
    }

    private static Polygon Arrowhead(double sx, double sy, double tx, double ty)
    {
        var dx = tx - sx;
        var dy = ty - sy;
        var len = Math.Sqrt(dx * dx + dy * dy);
        if (len < 0.0001) len = 1;
        var ux = dx / len;
        var uy = dy / len;
        // Perpendicular
        var px = -uy;
        var py = ux;

        var tip = new Point(tx, ty);
        var baseCenter = new Point(tx - ux * ArrowheadLength, ty - uy * ArrowheadLength);
        var left = new Point(baseCenter.X + px * ArrowheadHalfWidth, baseCenter.Y + py * ArrowheadHalfWidth);
        var right = new Point(baseCenter.X - px * ArrowheadHalfWidth, baseCenter.Y - py * ArrowheadHalfWidth);

        var poly = new Polygon
        {
            Points = new Points { tip, left, right },
            StrokeThickness = 0,
        };
        poly[!Shape.FillProperty] = new DynamicResourceExtension("Foreground");
        return poly;
    }
}
