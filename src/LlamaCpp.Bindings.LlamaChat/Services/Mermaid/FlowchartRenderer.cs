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
        foreach (var e in laid.Edges) AddEdge(canvas, e);
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

    private static void AddEdge(Canvas canvas, LaidOutEdge e)
    {
        if (e.Points.Length < 2) return;

        var (sx, sy) = e.Points[0];
        var (tx, ty) = e.Points[^1];

        var line = new Line
        {
            StartPoint = new Point(sx, sy),
            EndPoint = new Point(tx, ty),
            StrokeThickness = e.Edge.Style == EdgeStyle.ThickArrow ? ThickStrokeThickness : EdgeStrokeThickness,
        };
        line[!Shape.StrokeProperty] = new DynamicResourceExtension("Foreground");
        if (e.Edge.Style == EdgeStyle.DottedArrow)
        {
            line.StrokeDashArray = new AvaloniaList<double> { 2, 3 };
        }
        canvas.Children.Add(line);

        // Arrowhead (only for arrow styles, not plain SolidLine).
        if (e.Edge.Style != EdgeStyle.SolidLine)
        {
            canvas.Children.Add(Arrowhead(sx, sy, tx, ty));
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
            var midX = (sx + tx) / 2.0 - label.DesiredSize.Width / 2.0;
            var midY = (sy + ty) / 2.0 - label.DesiredSize.Height / 2.0;
            Canvas.SetLeft(label, midX);
            Canvas.SetTop(label, midY);
            canvas.Children.Add(label);
        }
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
