using System.Security.Cryptography;
using Microsoft.Data.Sqlite;

namespace LlamaCpp.Bindings;

/// <summary>
/// Append-only SQLite store for per-measurement ablation results.
/// Accumulates across builder runs and across models so a long-running
/// investigation (days/weeks) doesn't lose data when individual
/// campaigns are cancelled, when llama.cpp is updated, or when the user
/// switches between targets.
/// </summary>
/// <remarks>
/// <para>
/// The schema treats one PPL measurement as one row. A row's natural
/// identity is (model, corpus, imatrix, context, target, ablation_type)
/// — re-running the exact same measurement is allowed (it adds another
/// sample, useful for variance estimation), but the builder's resume
/// path uses this composite to skip cells that already have at least
/// one sample.
/// </para>
/// <para>
/// Identity strings (<c>model_sha</c>, <c>corpus_sha</c>,
/// <c>imatrix_sha</c>) are content-derived rather than path-based so a
/// renamed or relocated file still matches its prior measurements. See
/// <see cref="ComputeContentSha"/> for the cheap-but-stable hash used
/// for multi-GB GGUFs.
/// </para>
/// </remarks>
public sealed class LlamaInvestigationDb : IDisposable
{
    /// <summary>Default DB path: <c>~/.cache/llama-investigation/measurements.sqlite</c>.</summary>
    public static string DefaultPath
    {
        get
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".cache", "llama-investigation", "measurements.sqlite");
        }
    }

    /// <summary>
    /// Sentinel imatrix SHA used when a measurement was taken without
    /// an imatrix. Constants like this avoid nullable-string churn at
    /// the schema layer; queries can match on the explicit value.
    /// </summary>
    public const string NoImatrixSha = "none";

    private readonly SqliteConnection _conn;

    private LlamaInvestigationDb(SqliteConnection conn)
    {
        _conn = conn;
    }

    /// <summary>Open or create the DB at <paramref name="path"/> (default: <see cref="DefaultPath"/>).</summary>
    public static LlamaInvestigationDb Open(string? path = null)
    {
        path ??= DefaultPath;
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var conn = new SqliteConnection($"Data Source={path}");
        conn.Open();

        // WAL keeps reads concurrent with the single writer (the
        // builder process); synchronous=NORMAL is the right tradeoff
        // for an append-only investigation log — we'd rather lose the
        // last unflushed row in a crash than slow every insert.
        using (var pragma = conn.CreateCommand())
        {
            pragma.CommandText = "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;";
            pragma.ExecuteNonQuery();
        }

        EnsureSchema(conn);
        return new LlamaInvestigationDb(conn);
    }

    private static void EnsureSchema(SqliteConnection conn)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_sha TEXT NOT NULL,
                arch_id TEXT NOT NULL,
                param_count INTEGER NOT NULL,
                corpus_sha TEXT NOT NULL,
                corpus_name TEXT,
                imatrix_sha TEXT NOT NULL,
                context_size INTEGER NOT NULL,
                ablation_target TEXT NOT NULL,
                ablation_type INTEGER NOT NULL,
                baseline_type INTEGER NOT NULL,
                baseline_ppl REAL NOT NULL,
                ablation_ppl REAL NOT NULL,
                delta_ppl REAL NOT NULL,
                measured_at_utc TEXT NOT NULL,
                builder_version TEXT,
                llama_cpp_version TEXT,
                gpu_model TEXT,
                notes TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_meas_model ON measurements(model_sha);
            CREATE INDEX IF NOT EXISTS idx_meas_arch ON measurements(arch_id);
            CREATE INDEX IF NOT EXISTS idx_meas_target ON measurements(ablation_target);
            CREATE INDEX IF NOT EXISTS idx_meas_compose ON measurements(
                model_sha, corpus_sha, imatrix_sha, context_size,
                ablation_target, ablation_type);
        ";
        cmd.ExecuteNonQuery();
    }

    /// <summary>Insert one measurement row. Idempotency is the caller's responsibility — duplicates are allowed (adds a sample).</summary>
    public long RecordMeasurement(LlamaMeasurementRecord r)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            INSERT INTO measurements (
                model_sha, arch_id, param_count, corpus_sha, corpus_name,
                imatrix_sha, context_size, ablation_target, ablation_type,
                baseline_type, baseline_ppl, ablation_ppl, delta_ppl,
                measured_at_utc, builder_version, llama_cpp_version,
                gpu_model, notes)
            VALUES (
                $model_sha, $arch_id, $param_count, $corpus_sha, $corpus_name,
                $imatrix_sha, $context_size, $ablation_target, $ablation_type,
                $baseline_type, $baseline_ppl, $ablation_ppl, $delta_ppl,
                $measured_at_utc, $builder_version, $llama_cpp_version,
                $gpu_model, $notes);
            SELECT last_insert_rowid();
        ";
        cmd.Parameters.AddWithValue("$model_sha", r.ModelSha);
        cmd.Parameters.AddWithValue("$arch_id", r.ArchId);
        cmd.Parameters.AddWithValue("$param_count", r.ParamCount);
        cmd.Parameters.AddWithValue("$corpus_sha", r.CorpusSha);
        cmd.Parameters.AddWithValue("$corpus_name", (object?)r.CorpusName ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$imatrix_sha", r.ImatrixSha);
        cmd.Parameters.AddWithValue("$context_size", r.ContextSize);
        cmd.Parameters.AddWithValue("$ablation_target", r.AblationTarget);
        cmd.Parameters.AddWithValue("$ablation_type", (int)r.AblationType);
        cmd.Parameters.AddWithValue("$baseline_type", (int)r.BaselineType);
        cmd.Parameters.AddWithValue("$baseline_ppl", r.BaselinePpl);
        cmd.Parameters.AddWithValue("$ablation_ppl", r.AblationPpl);
        cmd.Parameters.AddWithValue("$delta_ppl", r.DeltaPpl);
        cmd.Parameters.AddWithValue("$measured_at_utc", r.MeasuredAtUtc.ToString("O"));
        cmd.Parameters.AddWithValue("$builder_version", (object?)r.BuilderVersion ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$llama_cpp_version", (object?)r.LlamaCppVersion ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$gpu_model", (object?)r.GpuModel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$notes", (object?)r.Notes ?? DBNull.Value);
        return (long)(cmd.ExecuteScalar() ?? 0L);
    }

    /// <summary>
    /// Query measurements matching <paramref name="filter"/>. Streams via yield-return
    /// so large result sets don't materialize all at once.
    /// </summary>
    public IEnumerable<LlamaMeasurementRecord> Query(LlamaMeasurementFilter filter)
    {
        using var cmd = _conn.CreateCommand();
        var sql = "SELECT model_sha, arch_id, param_count, corpus_sha, corpus_name, " +
                  "imatrix_sha, context_size, ablation_target, ablation_type, " +
                  "baseline_type, baseline_ppl, ablation_ppl, delta_ppl, " +
                  "measured_at_utc, builder_version, llama_cpp_version, gpu_model, notes " +
                  "FROM measurements WHERE 1=1";
        if (filter.ModelSha is not null)
        {
            sql += " AND model_sha = $model_sha";
            cmd.Parameters.AddWithValue("$model_sha", filter.ModelSha);
        }
        if (filter.ArchId is not null)
        {
            sql += " AND arch_id = $arch_id";
            cmd.Parameters.AddWithValue("$arch_id", filter.ArchId);
        }
        if (filter.CorpusSha is not null)
        {
            sql += " AND corpus_sha = $corpus_sha";
            cmd.Parameters.AddWithValue("$corpus_sha", filter.CorpusSha);
        }
        if (filter.ImatrixSha is not null)
        {
            sql += " AND imatrix_sha = $imatrix_sha";
            cmd.Parameters.AddWithValue("$imatrix_sha", filter.ImatrixSha);
        }
        if (filter.ContextSize is int cs)
        {
            sql += " AND context_size = $context_size";
            cmd.Parameters.AddWithValue("$context_size", cs);
        }
        if (filter.AblationTargetPrefix is not null)
        {
            sql += " AND ablation_target LIKE $target_prefix";
            cmd.Parameters.AddWithValue("$target_prefix", filter.AblationTargetPrefix + "%");
        }
        if (filter.AblationType is LlamaTensorType at)
        {
            sql += " AND ablation_type = $ablation_type";
            cmd.Parameters.AddWithValue("$ablation_type", (int)at);
        }
        sql += " ORDER BY measured_at_utc DESC";

        cmd.CommandText = sql;
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            yield return new LlamaMeasurementRecord(
                ModelSha:        reader.GetString(0),
                ArchId:          reader.GetString(1),
                ParamCount:      reader.GetInt64(2),
                CorpusSha:       reader.GetString(3),
                CorpusName:      reader.IsDBNull(4) ? null : reader.GetString(4),
                ImatrixSha:      reader.GetString(5),
                ContextSize:     reader.GetInt32(6),
                AblationTarget:  reader.GetString(7),
                AblationType:    (LlamaTensorType)reader.GetInt32(8),
                BaselineType:    (LlamaTensorType)reader.GetInt32(9),
                BaselinePpl:     reader.GetDouble(10),
                AblationPpl:     reader.GetDouble(11),
                DeltaPpl:        reader.GetDouble(12),
                MeasuredAtUtc:   DateTime.Parse(reader.GetString(13), null, System.Globalization.DateTimeStyles.RoundtripKind),
                BuilderVersion:  reader.IsDBNull(14) ? null : reader.GetString(14),
                LlamaCppVersion: reader.IsDBNull(15) ? null : reader.GetString(15),
                GpuModel:        reader.IsDBNull(16) ? null : reader.GetString(16),
                Notes:           reader.IsDBNull(17) ? null : reader.GetString(17));
        }
    }

    /// <summary>
    /// Has at least one measurement been taken at this exact (model, corpus,
    /// imatrix, context, target, type) combination? Drives the builder's
    /// "skip cells already in the DB" resume path.
    /// </summary>
    public bool HasMeasurement(
        string modelSha, string corpusSha, string imatrixSha,
        int contextSize, string ablationTarget, LlamaTensorType ablationType)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            SELECT 1 FROM measurements
            WHERE model_sha = $model_sha
              AND corpus_sha = $corpus_sha
              AND imatrix_sha = $imatrix_sha
              AND context_size = $context_size
              AND ablation_target = $ablation_target
              AND ablation_type = $ablation_type
            LIMIT 1;";
        cmd.Parameters.AddWithValue("$model_sha", modelSha);
        cmd.Parameters.AddWithValue("$corpus_sha", corpusSha);
        cmd.Parameters.AddWithValue("$imatrix_sha", imatrixSha);
        cmd.Parameters.AddWithValue("$context_size", contextSize);
        cmd.Parameters.AddWithValue("$ablation_target", ablationTarget);
        cmd.Parameters.AddWithValue("$ablation_type", (int)ablationType);
        return cmd.ExecuteScalar() is not null;
    }

    /// <summary>Total row count — for status displays.</summary>
    public long Count()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM measurements;";
        return (long)(cmd.ExecuteScalar() ?? 0L);
    }

    /// <summary>
    /// Count measurements matching a campaign signature, optionally
    /// filtered to specific ablation targets and/or types. Used by the
    /// UI's invalidation controls to preview "X rows would be deleted"
    /// before committing to a delete.
    /// </summary>
    public long CountMatching(
        string modelSha,
        string corpusSha,
        string imatrixSha,
        int contextSize,
        IReadOnlyList<string>? ablationTargets = null,
        IReadOnlyList<LlamaTensorType>? ablationTypes = null)
    {
        using var cmd = _conn.CreateCommand();
        var sql = "SELECT COUNT(*) FROM measurements WHERE " +
                  "model_sha = $model AND corpus_sha = $corpus AND " +
                  "imatrix_sha = $imatrix AND context_size = $ctx";
        cmd.Parameters.AddWithValue("$model", modelSha);
        cmd.Parameters.AddWithValue("$corpus", corpusSha);
        cmd.Parameters.AddWithValue("$imatrix", imatrixSha);
        cmd.Parameters.AddWithValue("$ctx", contextSize);
        AppendTargetTypeWhere(cmd, ref sql, ablationTargets, ablationTypes);
        cmd.CommandText = sql;
        return (long)(cmd.ExecuteScalar() ?? 0L);
    }

    /// <summary>
    /// Delete measurements matching a campaign signature, optionally
    /// filtered to specific ablation targets and/or types. Returns the
    /// number of rows removed. Useful for invalidating bogus rows after
    /// a builder bug fix (Run 14-style demote-drop bug) or for forcing a
    /// re-measurement of a specific cell with refreshed conditions
    /// (e.g. after a llama.cpp version bump that changed quantization
    /// tables).
    /// </summary>
    /// <remarks>
    /// Pass <c>null</c> for <paramref name="ablationTargets"/> or
    /// <paramref name="ablationTypes"/> to leave that dimension
    /// unconstrained. Passing both null deletes every row for the
    /// campaign signature.
    /// </remarks>
    public int DeleteMatching(
        string modelSha,
        string corpusSha,
        string imatrixSha,
        int contextSize,
        IReadOnlyList<string>? ablationTargets = null,
        IReadOnlyList<LlamaTensorType>? ablationTypes = null)
    {
        using var cmd = _conn.CreateCommand();
        var sql = "DELETE FROM measurements WHERE " +
                  "model_sha = $model AND corpus_sha = $corpus AND " +
                  "imatrix_sha = $imatrix AND context_size = $ctx";
        cmd.Parameters.AddWithValue("$model", modelSha);
        cmd.Parameters.AddWithValue("$corpus", corpusSha);
        cmd.Parameters.AddWithValue("$imatrix", imatrixSha);
        cmd.Parameters.AddWithValue("$ctx", contextSize);
        AppendTargetTypeWhere(cmd, ref sql, ablationTargets, ablationTypes);
        cmd.CommandText = sql;
        return cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Append <c>AND ablation_target IN (...)</c> and <c>AND ablation_type IN (...)</c>
    /// clauses to the running SQL when filters are present. Parameterized
    /// to keep target names and type ints out of the SQL text proper.
    /// </summary>
    private static void AppendTargetTypeWhere(
        SqliteCommand cmd, ref string sql,
        IReadOnlyList<string>? ablationTargets,
        IReadOnlyList<LlamaTensorType>? ablationTypes)
    {
        if (ablationTargets is { Count: > 0 })
        {
            var names = new List<string>(ablationTargets.Count);
            for (int i = 0; i < ablationTargets.Count; i++)
            {
                var p = $"$tgt{i}";
                names.Add(p);
                cmd.Parameters.AddWithValue(p, ablationTargets[i]);
            }
            sql += $" AND ablation_target IN ({string.Join(", ", names)})";
        }
        if (ablationTypes is { Count: > 0 })
        {
            var names = new List<string>(ablationTypes.Count);
            for (int i = 0; i < ablationTypes.Count; i++)
            {
                var p = $"$typ{i}";
                names.Add(p);
                cmd.Parameters.AddWithValue(p, (int)ablationTypes[i]);
            }
            sql += $" AND ablation_type IN ({string.Join(", ", names)})";
        }
    }

    public void Dispose()
    {
        _conn.Dispose();
    }

    // ------------------------------------------------------------------
    // Content-hash helpers — content-stable identity for files of any size.
    // ------------------------------------------------------------------

    /// <summary>
    /// Cheap, content-stable identifier: SHA-256 of (file_size ‖ first 1MB
    /// ‖ last 1MB). On a multi-GB GGUF this hashes ~2 MB instead of the
    /// whole file, so it runs in milliseconds. The leading length prefix
    /// guarantees that two files of different lengths always disagree
    /// even when their head/tail samples coincide.
    /// </summary>
    /// <remarks>
    /// Why not full-file SHA-256: hashing 30 GB blocks the UI for ~30 s.
    /// Why not (size + path): renaming or relocating a file would
    /// invalidate prior measurements — exactly what we don't want for a
    /// multi-week investigation database. The head+tail sample catches
    /// any meaningful content change while keeping the operation
    /// effectively free.
    /// </remarks>
    public static string ComputeContentSha(string filePath)
    {
        const long sampleSize = 1L * 1024 * 1024;    // 1 MiB head + 1 MiB tail

        using var fs = File.OpenRead(filePath);
        long total = fs.Length;
        using var sha = SHA256.Create();

        Span<byte> lengthBytes = stackalloc byte[8];
        BitConverter.TryWriteBytes(lengthBytes, total);
        sha.TransformBlock(lengthBytes.ToArray(), 0, lengthBytes.Length, null, 0);

        if (total <= sampleSize * 2)
        {
            // Small file — hash everything. Cuts straight through the
            // edge case where head and tail overlap on a < 2MB file.
            var all = new byte[total];
            int read = 0;
            while (read < all.Length)
            {
                int n = fs.Read(all, read, all.Length - read);
                if (n <= 0) break;
                read += n;
            }
            sha.TransformFinalBlock(all, 0, read);
        }
        else
        {
            var head = new byte[sampleSize];
            ReadFully(fs, head);
            sha.TransformBlock(head, 0, head.Length, null, 0);

            fs.Seek(total - sampleSize, SeekOrigin.Begin);
            var tail = new byte[sampleSize];
            ReadFully(fs, tail);
            sha.TransformFinalBlock(tail, 0, tail.Length);
        }

        return Convert.ToHexString(sha.Hash!).ToLowerInvariant();
    }

    /// <summary>SHA-256 of a UTF-8 text — used for corpus identity (corpora are typically MBs, full hash is fine).</summary>
    public static string ComputeTextSha(string text)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static void ReadFully(Stream s, byte[] buf)
    {
        int read = 0;
        while (read < buf.Length)
        {
            int n = s.Read(buf, read, buf.Length - read);
            if (n <= 0) throw new EndOfStreamException();
            read += n;
        }
    }
}

/// <summary>One ablation PPL measurement. See <see cref="LlamaInvestigationDb"/>.</summary>
public sealed record LlamaMeasurementRecord(
    string ModelSha,
    string ArchId,
    long ParamCount,
    string CorpusSha,
    string? CorpusName,
    string ImatrixSha,
    int ContextSize,
    /// <summary>
    /// Either <c>category:&lt;name&gt;</c> for whole-category ablations or
    /// <c>tensor:&lt;full_name&gt;</c> for per-layer measurements. The
    /// prefix lets queries cheaply distinguish modes.
    /// </summary>
    string AblationTarget,
    LlamaTensorType AblationType,
    LlamaTensorType BaselineType,
    double BaselinePpl,
    double AblationPpl,
    double DeltaPpl,
    DateTime MeasuredAtUtc,
    string? BuilderVersion,
    string? LlamaCppVersion,
    string? GpuModel,
    string? Notes);

/// <summary>Filter for <see cref="LlamaInvestigationDb.Query"/>. Unset fields are ignored.</summary>
public sealed class LlamaMeasurementFilter
{
    public string? ModelSha { get; set; }
    public string? ArchId { get; set; }
    public string? CorpusSha { get; set; }
    public string? ImatrixSha { get; set; }
    public int? ContextSize { get; set; }
    /// <summary>"category:" or "tensor:" to scope to one ablation mode.</summary>
    public string? AblationTargetPrefix { get; set; }
    public LlamaTensorType? AblationType { get; set; }
}
