using CommunityToolkit.Mvvm.ComponentModel;
using LlamaCpp.Bindings.LlamaChat.Models;

namespace LlamaCpp.Bindings.LlamaChat.ViewModels;

/// <summary>
/// Editable copy of <see cref="SamplerSettings"/> plus a few generation knobs.
/// NumericUpDown binds to decimal; we cast at snapshot time.
/// </summary>
public partial class SamplerPanelViewModel : ObservableObject
{
    // --- Core ---
    [ObservableProperty] private decimal _temperature = 0.7m;
    [ObservableProperty] private decimal _dynaTempRange = 0m;
    [ObservableProperty] private decimal _dynaTempExponent = 1m;
    [ObservableProperty] private decimal _seed = 0xDEADBEEFu;

    // --- Truncation (each has an enable flag for the null/disabled semantics) ---
    [ObservableProperty] private bool _topKEnabled = true;
    [ObservableProperty] private decimal _topK = 40m;
    [ObservableProperty] private bool _topPEnabled = true;
    [ObservableProperty] private decimal _topP = 0.95m;
    [ObservableProperty] private bool _minPEnabled = true;
    [ObservableProperty] private decimal _minP = 0.05m;
    [ObservableProperty] private bool _typicalEnabled = false;
    [ObservableProperty] private decimal _typical = 1.0m;
    [ObservableProperty] private bool _topNSigmaEnabled = false;
    [ObservableProperty] private decimal _topNSigma = 1.0m;

    // --- XTC ---
    [ObservableProperty] private bool _xtcEnabled = false;
    [ObservableProperty] private decimal _xtcProbability = 0.0m;
    [ObservableProperty] private decimal _xtcThreshold = 0.1m;

    // --- DRY ---
    [ObservableProperty] private decimal _dryMultiplier = 0m;
    [ObservableProperty] private decimal _dryBase = 1.75m;
    [ObservableProperty] private decimal _dryAllowedLength = 2m;
    [ObservableProperty] private decimal _dryPenaltyLastN = -1m;

    // --- Penalties ---
    [ObservableProperty] private decimal _penaltyLastN = 64m;
    [ObservableProperty] private decimal _penaltyRepeat = 1.0m;
    [ObservableProperty] private decimal _penaltyFrequency = 0.0m;
    [ObservableProperty] private decimal _penaltyPresence = 0.0m;

    // --- Terminal sampler ---
    [ObservableProperty] private MirostatMode _mirostat = MirostatMode.Off;
    [ObservableProperty] private decimal _mirostatTau = 5m;
    [ObservableProperty] private decimal _mirostatEta = 0.1m;

    // --- Grammar ---
    [ObservableProperty] private string _grammar = string.Empty;

    // --- Generation ---
    [ObservableProperty] private decimal _maxTokens = 1024m;
    [ObservableProperty] private bool _extractReasoning = true;
    [ObservableProperty] private bool _extractAsrTranscript = false;

    public SamplerSettings SnapshotSampler() => new()
    {
        Temperature = (float)Temperature,
        DynaTempRange = (float)DynaTempRange,
        DynaTempExponent = (float)DynaTempExponent,
        Seed = (uint)Seed,
        TopK = TopKEnabled ? (int)TopK : null,
        TopP = TopPEnabled ? (float)TopP : null,
        MinP = MinPEnabled ? (float)MinP : null,
        Typical = TypicalEnabled ? (float)Typical : null,
        TopNSigma = TopNSigmaEnabled ? (float)TopNSigma : null,
        XtcProbability = XtcEnabled ? (float)XtcProbability : null,
        XtcThreshold = (float)XtcThreshold,
        DryMultiplier = (float)DryMultiplier,
        DryBase = (float)DryBase,
        DryAllowedLength = (int)DryAllowedLength,
        DryPenaltyLastN = (int)DryPenaltyLastN,
        PenaltyLastN = (int)PenaltyLastN,
        PenaltyRepeat = (float)PenaltyRepeat,
        PenaltyFrequency = (float)PenaltyFrequency,
        PenaltyPresence = (float)PenaltyPresence,
        Mirostat = Mirostat,
        MirostatTau = (float)MirostatTau,
        MirostatEta = (float)MirostatEta,
        GbnfGrammar = string.IsNullOrWhiteSpace(Grammar) ? null : Grammar,
    };

    public GenerationSettings SnapshotGeneration() => new()
    {
        MaxTokens = (int)MaxTokens,
        ExtractReasoning = ExtractReasoning,
        ExtractAsrTranscript = ExtractAsrTranscript,
    };

    public void LoadFrom(SamplerSettings s, GenerationSettings g)
    {
        Temperature = (decimal)s.Temperature;
        DynaTempRange = (decimal)s.DynaTempRange;
        DynaTempExponent = (decimal)s.DynaTempExponent;
        Seed = s.Seed;
        TopKEnabled = s.TopK is not null;
        TopK = (decimal)(s.TopK ?? 40);
        TopPEnabled = s.TopP is not null;
        TopP = (decimal)(s.TopP ?? 0.95f);
        MinPEnabled = s.MinP is not null;
        MinP = (decimal)(s.MinP ?? 0.05f);
        TypicalEnabled = s.Typical is not null;
        Typical = (decimal)(s.Typical ?? 1.0f);
        TopNSigmaEnabled = s.TopNSigma is not null;
        TopNSigma = (decimal)(s.TopNSigma ?? 1.0f);
        XtcEnabled = s.XtcProbability is not null;
        XtcProbability = (decimal)(s.XtcProbability ?? 0f);
        XtcThreshold = (decimal)s.XtcThreshold;
        DryMultiplier = (decimal)s.DryMultiplier;
        DryBase = (decimal)s.DryBase;
        DryAllowedLength = s.DryAllowedLength;
        DryPenaltyLastN = s.DryPenaltyLastN;
        PenaltyLastN = s.PenaltyLastN;
        PenaltyRepeat = (decimal)s.PenaltyRepeat;
        PenaltyFrequency = (decimal)s.PenaltyFrequency;
        PenaltyPresence = (decimal)s.PenaltyPresence;
        Mirostat = s.Mirostat;
        MirostatTau = (decimal)s.MirostatTau;
        MirostatEta = (decimal)s.MirostatEta;
        Grammar = s.GbnfGrammar ?? string.Empty;
        MaxTokens = g.MaxTokens;
        ExtractReasoning = g.ExtractReasoning;
        ExtractAsrTranscript = g.ExtractAsrTranscript;
    }
}
