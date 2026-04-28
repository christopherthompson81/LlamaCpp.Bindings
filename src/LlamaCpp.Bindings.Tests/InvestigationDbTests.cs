namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Schema, round-trip, dedup, and content-hash behavior of
/// <see cref="LlamaInvestigationDb"/>. The DB is the persistent
/// substrate for multi-week investigations — anything wrong here
/// silently corrupts a user's accumulated work.
/// </summary>
public class InvestigationDbTests
{
    private static string TempDbPath() =>
        Path.Combine(Path.GetTempPath(), $"llama-invdb-{Guid.NewGuid():N}.sqlite");

    private static LlamaMeasurementRecord SampleRecord(
        string target = "category:ffn_up",
        LlamaTensorType type = LlamaTensorType.Q4_K,
        double deltaPpl = 0.123) =>
        new(
            ModelSha:        "model_abc",
            ArchId:          "qwen3",
            ParamCount:      1_000_000_000,
            CorpusSha:       "corpus_def",
            CorpusName:      "wiki.test.raw",
            ImatrixSha:      LlamaInvestigationDb.NoImatrixSha,
            ContextSize:     512,
            AblationTarget:  target,
            AblationType:    type,
            BaselineType:    LlamaTensorType.F16,
            BaselinePpl:     14.084,
            AblationPpl:     14.084 + deltaPpl,
            DeltaPpl:        deltaPpl,
            MeasuredAtUtc:   DateTime.UtcNow,
            BuilderVersion:  "test/1.0",
            LlamaCppVersion: "b8875",
            GpuModel:        "RTX 3090",
            Notes:           null);

    [Fact]
    public void RoundTrip_InsertAndQuery_ReturnsExactRecord()
    {
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            var rec = SampleRecord();
            var id = db.RecordMeasurement(rec);
            Assert.True(id > 0);

            var rows = db.Query(new LlamaMeasurementFilter { ModelSha = "model_abc" }).ToList();
            Assert.Single(rows);
            var got = rows[0];
            Assert.Equal(rec.ModelSha, got.ModelSha);
            Assert.Equal(rec.AblationTarget, got.AblationTarget);
            Assert.Equal(rec.AblationType, got.AblationType);
            Assert.Equal(rec.DeltaPpl, got.DeltaPpl, precision: 6);
            Assert.Equal(rec.GpuModel, got.GpuModel);
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void HasMeasurement_ChecksExactComposite()
    {
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            db.RecordMeasurement(SampleRecord("category:ffn_up", LlamaTensorType.Q4_K));

            // Exact match → present.
            Assert.True(db.HasMeasurement(
                "model_abc", "corpus_def", LlamaInvestigationDb.NoImatrixSha,
                512, "category:ffn_up", LlamaTensorType.Q4_K));

            // Different target → not present.
            Assert.False(db.HasMeasurement(
                "model_abc", "corpus_def", LlamaInvestigationDb.NoImatrixSha,
                512, "category:ffn_down", LlamaTensorType.Q4_K));

            // Different type → not present.
            Assert.False(db.HasMeasurement(
                "model_abc", "corpus_def", LlamaInvestigationDb.NoImatrixSha,
                512, "category:ffn_up", LlamaTensorType.Q2_K));

            // Different context → not present (this is intentional —
            // changing ctx changes the absolute PPL scale).
            Assert.False(db.HasMeasurement(
                "model_abc", "corpus_def", LlamaInvestigationDb.NoImatrixSha,
                1024, "category:ffn_up", LlamaTensorType.Q4_K));
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void Insert_DuplicateAllowed_AccumulatesSamples()
    {
        // Re-running a measurement is intentionally allowed — accumulating
        // duplicates lets the analyst estimate measurement variance from
        // the spread of repeated samples. Idempotency is the builder's
        // responsibility (via HasMeasurement), not a constraint at the
        // schema level.
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            db.RecordMeasurement(SampleRecord(deltaPpl: 0.10));
            db.RecordMeasurement(SampleRecord(deltaPpl: 0.12));
            db.RecordMeasurement(SampleRecord(deltaPpl: 0.11));

            var rows = db.Query(new LlamaMeasurementFilter()).ToList();
            Assert.Equal(3, rows.Count);
            Assert.Equal(3, db.Count());
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void Filter_AblationTargetPrefix_SeparatesCategoryAndTensorRows()
    {
        var path = TempDbPath();
        try
        {
            using var db = LlamaInvestigationDb.Open(path);
            db.RecordMeasurement(SampleRecord("category:ffn_up"));
            db.RecordMeasurement(SampleRecord("category:ffn_down"));
            db.RecordMeasurement(SampleRecord("tensor:blk.0.attn_q.weight"));
            db.RecordMeasurement(SampleRecord("tensor:blk.1.attn_q.weight"));

            var cats = db.Query(new LlamaMeasurementFilter { AblationTargetPrefix = "category:" }).ToList();
            Assert.Equal(2, cats.Count);
            Assert.All(cats, r => Assert.StartsWith("category:", r.AblationTarget));

            var tensors = db.Query(new LlamaMeasurementFilter { AblationTargetPrefix = "tensor:" }).ToList();
            Assert.Equal(2, tensors.Count);
            Assert.All(tensors, r => Assert.StartsWith("tensor:", r.AblationTarget));
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void Persistence_DataSurvivesReopen()
    {
        var path = TempDbPath();
        try
        {
            using (var db = LlamaInvestigationDb.Open(path))
                db.RecordMeasurement(SampleRecord());

            // Reopen — persistence is the whole point of this layer.
            using var db2 = LlamaInvestigationDb.Open(path);
            Assert.Equal(1, db2.Count());
        }
        finally { TryDelete(path); }
    }

    [Fact]
    public void ContentSha_TwoIdenticalFilesProduceSameHash()
    {
        var a = Path.GetTempFileName();
        var b = Path.GetTempFileName();
        try
        {
            var bytes = new byte[3 * 1024 * 1024];    // 3 MB — exercises both head and tail samples
            new Random(42).NextBytes(bytes);
            File.WriteAllBytes(a, bytes);
            File.WriteAllBytes(b, bytes);

            Assert.Equal(
                LlamaInvestigationDb.ComputeContentSha(a),
                LlamaInvestigationDb.ComputeContentSha(b));
        }
        finally { TryDelete(a); TryDelete(b); }
    }

    [Fact]
    public void ContentSha_DifferentMiddleBytesNotDetected_ButSizeAndEdgesAre()
    {
        // Honest documentation of the tradeoff: the content-sha samples
        // head + tail. A change confined to the middle of a multi-MB
        // file is NOT detected by this hash. This is the cost of being
        // fast on multi-GB GGUFs. The use case (renamed/relocated file
        // identity) doesn't need full-file integrity — only stability
        // under metadata-only operations.
        var a = Path.GetTempFileName();
        var b = Path.GetTempFileName();
        try
        {
            var ba = new byte[3 * 1024 * 1024];
            new Random(42).NextBytes(ba);
            var bb = (byte[])ba.Clone();
            // Flip bytes in the middle (outside the head/tail sample windows).
            int mid = ba.Length / 2;
            for (int i = 0; i < 1024; i++) bb[mid + i] ^= 0xFF;

            File.WriteAllBytes(a, ba);
            File.WriteAllBytes(b, bb);

            // Same size + same head/tail samples → same hash even though
            // contents differ. Documented as expected behavior.
            Assert.Equal(
                LlamaInvestigationDb.ComputeContentSha(a),
                LlamaInvestigationDb.ComputeContentSha(b));

            // But size changes always produce a different hash (size is
            // hashed first, before any sample bytes).
            File.WriteAllBytes(b, ba.Take(ba.Length - 1).ToArray());
            Assert.NotEqual(
                LlamaInvestigationDb.ComputeContentSha(a),
                LlamaInvestigationDb.ComputeContentSha(b));
        }
        finally { TryDelete(a); TryDelete(b); }
    }

    [Fact]
    public void TextSha_StableAndDistinctForDifferentInputs()
    {
        var h1 = LlamaInvestigationDb.ComputeTextSha("hello world");
        var h2 = LlamaInvestigationDb.ComputeTextSha("hello world");
        var h3 = LlamaInvestigationDb.ComputeTextSha("hello world ");    // trailing space
        Assert.Equal(h1, h2);
        Assert.NotEqual(h1, h3);
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best-effort */ }
        // SQLite WAL journal/shm sidecars
        try { if (File.Exists(path + "-wal")) File.Delete(path + "-wal"); }
        catch { /* best-effort */ }
        try { if (File.Exists(path + "-shm")) File.Delete(path + "-shm"); }
        catch { /* best-effort */ }
    }
}
