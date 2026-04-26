using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace LlamaCpp.Bindings.GGUFLab.ViewModels;

/// <summary>
/// Drives the GGUF Editor page: open a GGUF, view all metadata KVs and
/// tensor info, edit/add/remove metadata, save to a new file.
/// </summary>
/// <remarks>
/// V1 scope is metadata-focused: tensor data is preserved byte-for-byte
/// during save (streamed from the source file). Tensor renaming and
/// re-quantization are explicitly out of scope — use the Quantize tool
/// for the latter.
/// </remarks>
public sealed partial class GgufEditorViewModel : ToolPageViewModel
{
    public override string Title => "GGUF Editor";
    public override string Description =>
        "Inspect and edit a GGUF's metadata. Tensor data is preserved byte-for-byte on save.";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLoaded))]
    private LlamaGgufFile? _file;

    public bool IsLoaded => File is not null;

    [ObservableProperty]
    private string _sourcePath = string.Empty;

    [ObservableProperty]
    private string _statusLine = "Open a GGUF to begin.";

    public ObservableCollection<MetadataRow> MetadataRows { get; } = new();
    public ObservableCollection<TensorRow> TensorRows { get; } = new();

    /// <summary>Selected metadata row — bound to the right-hand edit panel.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasMetadataSelection))]
    [NotifyPropertyChangedFor(nameof(EditValueTypeName))]
    private MetadataRow? _selectedMetadataRow;

    public bool HasMetadataSelection => SelectedMetadataRow is not null;

    public string EditValueTypeName =>
        SelectedMetadataRow is null
            ? string.Empty
            : SelectedMetadataRow.Entry.Value.Type == LlamaGgufType.Array
                ? $"Array<{SelectedMetadataRow.Entry.Value.InnerType}>"
                : SelectedMetadataRow.Entry.Value.Type.ToString();

    /// <summary>
    /// Editable text rendering of the selected value. Strings are shown
    /// raw; scalars use invariant culture; arrays are read-only with a
    /// summary in <see cref="EditValueText"/>.
    /// </summary>
    [ObservableProperty]
    private string _editValueText = string.Empty;

    /// <summary>Whether the current selection's type is editable in V1.</summary>
    public bool EditValueIsEditable =>
        SelectedMetadataRow is not null &&
        SelectedMetadataRow.Entry.Value.Type != LlamaGgufType.Array;

    /// <summary>Inline status / error from the last edit attempt.</summary>
    [ObservableProperty]
    private string _editStatusLine = string.Empty;

    // --- Add-metadata fields ---

    [ObservableProperty]
    private string _newMetadataKey = string.Empty;

    [ObservableProperty]
    private LlamaGgufType _newMetadataType = LlamaGgufType.String;

    public IReadOnlyList<LlamaGgufType> AddableMetadataTypes { get; } = new[]
    {
        LlamaGgufType.String,
        LlamaGgufType.Uint32,
        LlamaGgufType.Int32,
        LlamaGgufType.Uint64,
        LlamaGgufType.Int64,
        LlamaGgufType.Float32,
        LlamaGgufType.Float64,
        LlamaGgufType.Bool,
    };

    [ObservableProperty]
    private string _newMetadataValueText = string.Empty;

    partial void OnSelectedMetadataRowChanged(MetadataRow? value)
    {
        EditValueText = value is null ? string.Empty : RenderEditableValue(value.Entry.Value);
        EditStatusLine = string.Empty;
        OnPropertyChanged(nameof(EditValueIsEditable));
    }

    /// <summary>Called from the view's code-behind once a path has been picked.</summary>
    public Task OpenAsync(string path)
    {
        try
        {
            var file = LlamaGgufFile.Open(path);
            File = file;
            SourcePath = path;
            ReloadRowsFromFile();
            StatusLine = $"Loaded {Path.GetFileName(path)} — {file.Metadata.Count:N0} metadata, {file.Tensors.Count:N0} tensors.";
        }
        catch (Exception ex)
        {
            StatusLine = $"Failed to open {Path.GetFileName(path)}: {ex.Message}";
        }
        return Task.CompletedTask;
    }

    /// <summary>Save the current (mutated) state to <paramref name="outputPath"/>.</summary>
    public async Task SaveAsync(string outputPath)
    {
        if (File is null)
        {
            StatusLine = "Open a file first.";
            return;
        }
        try
        {
            StatusLine = "Saving…";
            await File.SaveAsAsync(outputPath);
            StatusLine = $"Saved → {Path.GetFileName(outputPath)} ({new FileInfo(outputPath).Length:N0} bytes).";
        }
        catch (Exception ex)
        {
            StatusLine = $"Save failed: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ApplyEdit()
    {
        if (SelectedMetadataRow is null)
        {
            EditStatusLine = "Select a row first.";
            return;
        }
        if (!EditValueIsEditable)
        {
            EditStatusLine = "Array editing is not supported in V1.";
            return;
        }

        try
        {
            var newValue = ParseEditableValue(SelectedMetadataRow.Entry.Value.Type, EditValueText);
            SelectedMetadataRow.Entry.Value = newValue;
            SelectedMetadataRow.RefreshDisplayCells();
            EditStatusLine = "Applied.";
        }
        catch (Exception ex)
        {
            EditStatusLine = $"Couldn't parse value: {ex.Message}";
        }
    }

    [RelayCommand]
    private void RemoveSelectedMetadata()
    {
        if (File is null || SelectedMetadataRow is null) return;
        var row = SelectedMetadataRow;
        File.Metadata.Remove(row.Entry);
        MetadataRows.Remove(row);
        SelectedMetadataRow = null;
        StatusLine = $"Removed metadata key '{row.Entry.Key}'. Save to persist.";
    }

    [RelayCommand]
    private void AddMetadata()
    {
        if (File is null)
        {
            StatusLine = "Open a file first.";
            return;
        }
        if (string.IsNullOrWhiteSpace(NewMetadataKey))
        {
            StatusLine = "New metadata needs a key.";
            return;
        }
        if (File.Metadata.Any(m => m.Key == NewMetadataKey))
        {
            StatusLine = $"Metadata key '{NewMetadataKey}' already exists.";
            return;
        }
        try
        {
            var value = ParseEditableValue(NewMetadataType, NewMetadataValueText);
            var entry = new LlamaGgufMetadataEntry(NewMetadataKey, value);
            File.Metadata.Add(entry);
            MetadataRows.Add(new MetadataRow(entry));
            StatusLine = $"Added metadata key '{NewMetadataKey}'.";
            NewMetadataKey = string.Empty;
            NewMetadataValueText = string.Empty;
        }
        catch (Exception ex)
        {
            StatusLine = $"Couldn't add: {ex.Message}";
        }
    }

    private void ReloadRowsFromFile()
    {
        MetadataRows.Clear();
        TensorRows.Clear();
        if (File is null) return;

        foreach (var m in File.Metadata)
        {
            MetadataRows.Add(new MetadataRow(m));
        }
        foreach (var t in File.Tensors)
        {
            TensorRows.Add(new TensorRow(t));
        }
    }

    private static string RenderEditableValue(LlamaGgufValue v) => v.Type switch
    {
        LlamaGgufType.String  => v.AsString(),
        LlamaGgufType.Bool    => v.AsBool() ? "true" : "false",
        LlamaGgufType.Uint8   => v.AsUInt8().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Int8    => v.AsInt8().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Uint16  => v.AsUInt16().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Int16   => v.AsInt16().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Uint32  => v.AsUInt32().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Int32   => v.AsInt32().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Float32 => v.AsFloat32().ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Uint64  => v.AsUInt64().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Int64   => v.AsInt64().ToString(System.Globalization.CultureInfo.InvariantCulture),
        LlamaGgufType.Float64 => v.AsFloat64().ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        _ => v.ToDisplayString(),
    };

    private static LlamaGgufValue ParseEditableValue(LlamaGgufType type, string text)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        return type switch
        {
            LlamaGgufType.String  => LlamaGgufValue.String(text ?? string.Empty),
            LlamaGgufType.Bool    => LlamaGgufValue.Bool(ParseBool(text)),
            LlamaGgufType.Uint8   => LlamaGgufValue.UInt8(byte.Parse(text, inv)),
            LlamaGgufType.Int8    => LlamaGgufValue.Int8(sbyte.Parse(text, inv)),
            LlamaGgufType.Uint16  => LlamaGgufValue.UInt16(ushort.Parse(text, inv)),
            LlamaGgufType.Int16   => LlamaGgufValue.Int16(short.Parse(text, inv)),
            LlamaGgufType.Uint32  => LlamaGgufValue.UInt32(uint.Parse(text, inv)),
            LlamaGgufType.Int32   => LlamaGgufValue.Int32(int.Parse(text, inv)),
            LlamaGgufType.Float32 => LlamaGgufValue.Float32(float.Parse(text, inv)),
            LlamaGgufType.Uint64  => LlamaGgufValue.UInt64(ulong.Parse(text, inv)),
            LlamaGgufType.Int64   => LlamaGgufValue.Int64(long.Parse(text, inv)),
            LlamaGgufType.Float64 => LlamaGgufValue.Float64(double.Parse(text, inv)),
            _ => throw new InvalidOperationException($"Type {type} not editable in V1.")
        };
    }

    private static bool ParseBool(string text)
    {
        text = text?.Trim().ToLowerInvariant() ?? "";
        if (text is "true" or "1" or "yes") return true;
        if (text is "false" or "0" or "no" or "") return false;
        throw new FormatException($"'{text}' is not a valid bool. Use true/false.");
    }
}

/// <summary>One row in the metadata table — wraps a mutable entry and exposes display cells.</summary>
public sealed partial class MetadataRow : ObservableObject
{
    public LlamaGgufMetadataEntry Entry { get; }

    [ObservableProperty]
    private string _typeName;

    [ObservableProperty]
    private string _displayValue;

    public string Key => Entry.Key;

    public MetadataRow(LlamaGgufMetadataEntry entry)
    {
        Entry = entry;
        _typeName = RenderTypeName(entry.Value);
        _displayValue = entry.Value.ToDisplayString();
    }

    public void RefreshDisplayCells()
    {
        TypeName = RenderTypeName(Entry.Value);
        DisplayValue = Entry.Value.ToDisplayString();
    }

    private static string RenderTypeName(LlamaGgufValue v) =>
        v.Type == LlamaGgufType.Array ? $"Array<{v.InnerType}>" : v.Type.ToString();
}

/// <summary>One row in the tensor table — read-only.</summary>
public sealed class TensorRow
{
    public string Name { get; }
    public string TypeName { get; }
    public string Shape { get; }
    public string ByteSize { get; }

    public TensorRow(LlamaGgufTensorInfo t)
    {
        Name = t.Name;
        TypeName = t.Type?.ToString() ?? $"type#{t.TypeId}";
        Shape = "[" + string.Join(", ", t.Dimensions) + "]";
        ByteSize = t.ByteSize.ToString("N0", System.Globalization.CultureInfo.InvariantCulture);
    }
}
