## Run 1 — 2026-04-23

**Question:** what's the path to KaTeX math + Mermaid diagram rendering in the LlamaChat Avalonia app, given we explicitly avoid WebView / heavy runtime deps?

**Scope for this doc:** the two deferred markdown features in `docs/webui_feature_checklist.md` §3 — KaTeX math (line 90) and Mermaid diagrams (line 99).

**Findings so far (library survey):**

- **LaTeX/math.**
  - `AvaloniaMath` 2.1.0 (xaml-math) — Avalonia-native `FormulaBlock`, MIT, July 2023. Hard-constrained to `Avalonia >=11.0 <12.0`. Upstream repo is active (commits yesterday) but still on `Avalonia 11.3.14` as of 2026-04-22. **Does not work on Avalonia 12.** Not an option without forking.
  - `CSharpMath.SkiaSharp` — 0.5.1 (May 2021) stable, `1.0.0-pre.1` (Feb 2026) prerelease. .NET Standard 2.0, MIT. Depends on `SkiaSharp >= 2.80.1`. We ship SkiaSharp `3.119.3-preview.1.1` transitively via Avalonia.Skia. SkiaSharp 2→3 had API changes (colour/paint) — 0.5.1 may break at runtime; 1.0.0-pre.1 likely rebuilt for 3.x.
  - Chosen: **CSharpMath.SkiaSharp 1.0.0-pre.1 first**, fall back to 0.5.1 if pre fails. Path: LaTeX → `SKBitmap` → PNG bytes → `Avalonia.Media.Imaging.Bitmap` → wrapped in `Image` control (block math) or `InlineUIContainer(Image)` (inline math).

- **Mermaid.**
  - No native .NET Mermaid renderer on NuGet. All "Mermaid" NuGet hits are source *generators* (emit mermaid syntax), not renderers.
  - WebView-based approaches (`Avalonia.WebView`, `WebViewControl.Avalonia`) rejected — adds ~100MB+ of runtime deps and is heavyweight for the use-case.
  - `Jint + Mermaid.js` infeasible (Mermaid requires DOM for label sizing).
  - `ClearScript V8 + Mermaid.js + svgdom shim` heavy and still weeks of work.
  - Kroki HTTP rejected — privacy regression for a local-LLM app.
  - **Chosen: hand-roll in pure C#.** Label sizing is measurable via Avalonia's `FormattedText`. The remaining cost is the layout algorithm + renderer.
  - V1 scope: flowchart only (`graph TD/LR/RL/BT`, `flowchart TD/...`), node shapes rectangle/rounded/stadium/circle/rhombus, edges `-->` / `---` / `-.->` / `==>` with optional labels, Sugiyama layered layout with median-heuristic crossing reduction, orthogonal edge routing.
  - Other graph-shaped diagram types (state, class, ER, C4, architecture) are flowchart-family — same layout engine, different node templates + edge decorators. Mindmap + gitgraph reuse the primitives with a different layout driver. Linear types (sequence, gantt, journey, timeline, kanban) each get their own lane-based renderer. Charts may lean on `LiveChartsCore` / `OxyPlot` rather than hand-roll.

**Plan for this run (KaTeX):**

1. Add `CSharpMath.SkiaSharp` 1.0.0-pre.1. Build. If package restore or runtime fails, fall back to 0.5.1 — document the failure mode here.
2. Add `.UseMathematics()` to the Markdig pipeline in `MarkdownRenderer`. It emits `MathInline` / `MathBlock` nodes.
3. New `Services/MathRenderer.cs`: LaTeX string → `Avalonia.Media.Imaging.Bitmap` via CSharpMath → `SKBitmap` → PNG stream → `Bitmap(Stream)`. Cache on `(latex, displayStyle, fontSize, foregroundHex)` keys since the same formula often repeats in chat.
4. Hook `MathBlock` in `RenderBlock` to emit an `Image` control (centred, sized to native bitmap dpi-1).
5. Hook `MathInline` in `AppendInline` to emit an `InlineUIContainer(Image)` with vertical alignment adjusted so the baseline roughly matches surrounding text.
6. Visual check with a handful of sample formulas: Euler's identity, a fraction, a summation, a matrix, inline `$x^2$` mid-paragraph.

## Run 2 — 2026-04-23 14:16

**Question:** does `CSharpMath.SkiaSharp 1.0.0-pre.1` actually work against the `SkiaSharp 3.119.3-preview` that Avalonia 12 transitively pulls in?

**Command:** dropped a throwaway console project into `/tmp/mathtest` that calls `new MathPainter { LaTeX = "..." }.DrawAsStream(SKEncodedImageFormat.Png)` for both display and inline styles. Wrote the display result to `out.png`.

**Finding:** works. Display mode returns a 1624-byte 175×62 RGBA PNG for `\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}`, visually correct. Inline mode likewise returns a non-null stream for `x^2 + y^2 = z^2`. No SkiaSharp 2→3 API break in the code path CSharpMath exercises.

**Surprises / API shape notes:**
- `MathPainter` has no `BackgroundColor` setter in the 1.0.0-pre.1 API — defaults to transparent, which is what we want anyway.
- Inline-style toggle is `LineStyle = CSharpMath.Atom.LineStyle.Text`, not a flag on `MathPainter` directly.
- `Application.TryFindResource(key, theme, out)` doesn't resolve as an extension on `Application` in Avalonia 12 the way it does on `IResourceHost`-implementing controls — use `app.Resources.TryGetResource(key, theme, out)` instead.
- `MathBlock` extends `FencedCodeBlock` in Markdig, so it must be arm-ordered *before* `FencedCodeBlock` in the render switch.

**Outcome:** KaTeX rendering is live in `MarkdownRenderer`. Block math emits a centred `Image` at display-style size 20; inline math emits an `InlineUIContainer(Image)` at text-style size 16 with `BaselineAlignment.Center`. Foreground colour is baked from the theme's `Foreground` brush at render time; cache key includes the ARGB so light/dark flips repopulate rather than show stale colour.

**Next:** move on to Mermaid (parser → Sugiyama layout → Avalonia canvas renderer). KaTeX item in `webui_feature_checklist.md` §3 flips to `[x]`.

## Run 3 — 2026-04-23 15:00

**Goal:** Mermaid flowchart v1 end-to-end. Scope per plan:

- Parser: `graph TD/LR/RL/BT`, `flowchart TD/LR/RL/BT` headers (header optional, defaults to `TD`); node shapes `A[rect]`, `A(rounded)`, `A([stadium])`, `A((circle))`, `A{rhombus}`; edges `-->`, `---`, `-.->`, `==>`; optional edge label `-->|text|`; node labels inside the shape delimiters. Multi-edge shorthand (`A --> B & C`, `A & B --> C`) deferred. Subgraphs (`subgraph name ... end`) deferred.
- Layout: Sugiyama layered. Cycle removal via DFS (reverse back-edges). Layer assignment via longest-path from sources. Crossing reduction via median heuristic, several downward/upward passes. Coordinate assignment: evenly spaced within each layer, layer gap = max node size + padding. Direction baked into layout axes (TD: layers run Y+, nodes within layer run X; LR: layers run X+, nodes within layer run Y).
- Edge routing v1: straight line segments with arrowheads. Orthogonal routing deferred — straight lines are acceptable for flowcharts up to ~20 nodes and hugely simpler to implement. Revisit if visual quality demands it.
- Renderer: Avalonia `Canvas`. Each node placed via `Canvas.SetLeft`/`SetTop`; shape drawn via `Rectangle`/`Ellipse`/`Path` with `DynamicResource Foreground`/`Muted` brushes; label `TextBlock` centred inside. Edges as `Polyline` + arrowhead `Polygon`. Label sizing measured via `FormattedText` with our `CodeFontFamily`... no, with the default app font, at 12px.

**File layout:**

```
Services/Mermaid/
  FlowchartGraph.cs     data model (POCO)
  FlowchartParser.cs    string → FlowchartGraph
  FlowchartLayout.cs    FlowchartGraph → LaidOutGraph (positions)
  FlowchartRenderer.cs  LaidOutGraph → Avalonia Control
```

And a top-level `Services/MermaidRenderer.cs` as the entrypoint used by `MarkdownRenderer` for `mermaid` fenced code blocks. Parser + layout are Avalonia-free and unit-testable; renderer is the only Avalonia-touching piece.

**Testability:** parser tests in `LlamaCpp.Bindings.Tests` are cheap (pure C#). Layout tests harder to assert — will verify by snapshotting node positions for a couple of reference graphs and manually reviewing the rendered output during dev.
