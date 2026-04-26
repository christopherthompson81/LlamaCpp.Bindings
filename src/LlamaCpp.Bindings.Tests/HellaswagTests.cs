namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Tests for <see cref="LlamaHellaswag"/>. Parser tests cover the
/// 6-line-per-task format and basic error reporting; the scoring
/// test runs a couple of hand-crafted tasks where the gold ending is
/// dramatically more likely than the alternatives — a competent base
/// model should pick at least one of two right.
/// </summary>
public class HellaswagTests
{
    [Fact]
    public void Parses_Six_Line_Per_Task_Format()
    {
        var text = string.Join("\n", new[]
        {
            "Activity: the dog ran",
            "0",
            " into the yard.",
            " across the moon.",
            " through the wormhole.",
            " toward the sky.",
            "Activity: she opened",
            "2",
            " the box.",
            " a red door.",
            " the door.",
            " a sigh.",
            "",   // trailing blank line, should be tolerated
        });

        var tasks = LlamaHellaswag.ParseUpstreamText(text);

        Assert.Equal(2, tasks.Count);
        Assert.Equal("Activity: the dog ran", tasks[0].Context);
        Assert.Equal(0, tasks[0].GoldEndingIndex);
        Assert.Equal(4, tasks[0].Endings.Count);
        Assert.Equal(" into the yard.", tasks[0].Endings[0]);
        Assert.Equal(2, tasks[1].GoldEndingIndex);
        Assert.Equal(" the door.", tasks[1].Endings[2]);
    }

    [Fact]
    public void Rejects_File_Where_Line_Count_Is_Not_Multiple_Of_Six()
    {
        var text = "ctx\n0\nA\nB\nC\n"; // 5 lines, not divisible by 6
        Assert.Throws<InvalidDataException>(() => LlamaHellaswag.ParseUpstreamText(text));
    }

    [Fact]
    public void Rejects_File_With_Non_Integer_Gold_Index()
    {
        var text = "ctx\nnot-a-number\nA\nB\nC\nD\n";
        Assert.Throws<InvalidDataException>(() => LlamaHellaswag.ParseUpstreamText(text));
    }

    [Fact]
    public async Task Scores_Two_Hand_Crafted_Tasks_Plausibly()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        // Two tasks where the "gold" ending is dramatically more
        // likely than the three obviously-wrong alternatives. A
        // competent base model should pick the correct ending on at
        // least one — we assert ≥1/2 to be robust to any single-task
        // weirdness without making the test a coin flip.
        var tasks = new[]
        {
            new LlamaHellaswagTask(
                Context: "The cat sat on the",
                Endings: new[]
                {
                    " mat and started to purr.",          // 0 — gold
                    " quasar and exploded violently.",    // 1
                    " purple cucumber-shaped recursion.", // 2
                    " orbital chemistry of mahogany.",    // 3
                },
                GoldEndingIndex: 0),
            new LlamaHellaswagTask(
                Context: "Once upon a time, there was a",
                Endings: new[]
                {
                    " quantum nematode of zealous fortnight.",
                    " little girl who lived in a small village.", // 1 — gold
                    " hyperbolic socket wrench tetrahedron.",
                    " noncommutative breakfast asymptote.",
                },
                GoldEndingIndex: 1),
        };

        var result = await LlamaHellaswag.ComputeAsync(model, tasks,
            new LlamaHellaswagOptions { ContextSize = 64 });

        Assert.Equal(2, result.TaskCount);
        Assert.True(result.CorrectNorm >= 1,
            $"Expected the model to pick the obvious ending on at least 1 of 2 hand-crafted tasks; got {result.CorrectNorm}/2.");
        Assert.InRange(result.AccuracyNorm, 0.0, 1.0);
    }

    [Fact]
    public async Task Honors_Cancellation_Between_Tasks()
    {
        var modelPath = TestModelProvider.EnsureModelPath();
        LlamaBackend.Initialize();

        using var model = new LlamaModel(modelPath, new LlamaModelParameters
        {
            GpuLayerCount = 0,
            UseMmap = true,
        });

        var tasks = new[]
        {
            new LlamaHellaswagTask("The dog ran", new[] { " home.", " away.", " up.", " down." }, 0),
        };

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            LlamaHellaswag.ComputeAsync(model, tasks, options: null,
                progress: null, cancellationToken: cts.Token));
    }
}
