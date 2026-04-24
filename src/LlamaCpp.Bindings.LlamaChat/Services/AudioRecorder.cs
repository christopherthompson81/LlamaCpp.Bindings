using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Silk.NET.OpenAL;
using Silk.NET.OpenAL.Extensions.EXT;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Cross-platform microphone capture via OpenAL Soft (through Silk.NET.OpenAL).
/// Captures mono 16-bit PCM at 16 kHz — the format Qwen3-ASR / Whisper accept
/// natively, so the WAV bytes we emit round-trip through the existing mtmd
/// audio-attachment path with no resampling.
///
/// Requires an OpenAL runtime at load time: shipped with most Linux distros
/// as <c>libopenal1</c>, built into macOS, and usually bundled as
/// <c>soft_oal.dll</c> on Windows. If the runtime is missing or no mic is
/// available, <see cref="Start"/> throws <see cref="InvalidOperationException"/>.
/// </summary>
public sealed class AudioRecorder : IDisposable
{
    public const int SampleRate = 16_000;

    // OpenAL internal ring-buffer size. At 16 kHz, 32 000 samples ≈ 2 s of
    // slack; any poll gap shorter than that won't drop samples.
    private const int CaptureRingSamples = 32_000;

    // We pull samples out of OpenAL in small chunks (~64 ms) on the poll
    // thread. Big enough to keep the syscall rate low, small enough that
    // Elapsed advances smoothly for the UI.
    private const int PollChunkSamples = 1024;

    // Hard cap so a forgotten recording can't grow indefinitely.
    private const int MaxSamples = SampleRate * 60 * 10; // 10 minutes

    private readonly List<short> _samples = new();
    private readonly object _lock = new();

    private CancellationTokenSource? _cts;
    private Task? _captureTask;
    private ALContext? _alc;
    private Capture? _capture;
    private unsafe Device* _device;

    public bool IsRecording { get; private set; }

    /// <summary>Duration captured so far, safe to read from the UI thread.</summary>
    public TimeSpan Elapsed
    {
        get
        {
            int count;
            lock (_lock) count = _samples.Count;
            return TimeSpan.FromSeconds((double)count / SampleRate);
        }
    }

    /// <summary>Open the default mic and begin capture. Idempotent (no-op if already recording).</summary>
    public unsafe void Start()
    {
        if (IsRecording) return;

        try
        {
            _alc = ALContext.GetApi();
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "OpenAL runtime not found. Install 'libopenal1' (Linux) or bundle 'soft_oal.dll' (Windows).",
                ex);
        }

        if (!_alc.TryGetExtension<Capture>(null, out var capture) || capture is null)
        {
            CleanupApis();
            throw new InvalidOperationException(
                "ALC_EXT_CAPTURE not supported by the installed OpenAL runtime.");
        }
        _capture = capture;

        _device = _capture.CaptureOpenDevice(null, (uint)SampleRate, BufferFormat.Mono16, CaptureRingSamples);
        if (_device == null)
        {
            CleanupApis();
            throw new InvalidOperationException(
                "Could not open the default microphone. Check OS permissions and that an input device is available.");
        }

        _capture.CaptureStart(_device);

        lock (_lock) _samples.Clear();
        _cts = new CancellationTokenSource();
        IsRecording = true;
        _captureTask = Task.Run(() => CaptureLoop(_cts.Token));
    }

    /// <summary>
    /// Stop capture and return the recorded samples (16-bit mono PCM at <see cref="SampleRate"/>).
    /// Returns empty if we weren't recording.
    /// </summary>
    public async Task<short[]> StopAsync()
    {
        if (!IsRecording) return Array.Empty<short>();

        _cts?.Cancel();
        try
        {
            if (_captureTask is not null) await _captureTask.ConfigureAwait(false);
        }
        catch
        {
            // Any error inside the loop is non-fatal here — we still want to
            // close the device and return what we captured.
        }
        IsRecording = false;

        CloseDevice();

        short[] result;
        lock (_lock) result = _samples.ToArray();
        return result;
    }

    private unsafe void CaptureLoop(CancellationToken ct)
    {
        var chunk = new short[PollChunkSamples];
        while (!ct.IsCancellationRequested && _device != null && _capture is not null && _alc is not null)
        {
            int available = 0;
            _alc.GetContextProperty(_device, (GetContextInteger)GetCaptureContextInteger.CaptureSamples, sizeof(int), (nint)(&available));

            if (available >= PollChunkSamples)
            {
                fixed (short* p = chunk)
                    _capture.CaptureSamples(_device, p, PollChunkSamples);

                lock (_lock)
                {
                    if (_samples.Count + PollChunkSamples <= MaxSamples)
                        _samples.AddRange(chunk);
                    else
                        return; // hit cap — stop accumulating; loop will exit
                }
            }
            else
            {
                Thread.Sleep(20);
            }
        }

        // Drain whatever is still in OpenAL's ring buffer at stop time.
        if (_device != null && _capture is not null && _alc is not null)
        {
            int remaining = 0;
            _alc.GetContextProperty(_device, (GetContextInteger)GetCaptureContextInteger.CaptureSamples, sizeof(int), (nint)(&remaining));
            if (remaining > 0)
            {
                var tail = new short[remaining];
                fixed (short* p = tail)
                    _capture.CaptureSamples(_device, p, remaining);
                lock (_lock)
                {
                    int room = MaxSamples - _samples.Count;
                    if (room > 0) _samples.AddRange(room >= remaining ? tail : tail.AsSpan(0, room).ToArray());
                }
            }
        }
    }

    private unsafe void CloseDevice()
    {
        if (_device != null && _capture is not null)
        {
            try { _capture.CaptureStop(_device); } catch { }
            try { _capture.CaptureCloseDevice(_device); } catch { }
            _device = null;
        }
    }

    // Deliberately does NOT dispose _alc / _capture: Silk.NET's native API
    // containers hold a shared process-wide handle to libopenal.so, and
    // disposing them unloads the library in a way that fails the second time
    // a recorder is opened (System.InvalidOperationException from
    // NativeLibrary.Free). The handles are cheap to keep alive for the
    // process lifetime.
    private void CleanupApis()
    {
        _capture = null;
        _alc = null;
    }

    public void Dispose()
    {
        try { _cts?.Cancel(); } catch { }
        try { _captureTask?.Wait(1_000); } catch { }
        CloseDevice();
        CleanupApis();
        _cts?.Dispose();
        _cts = null;
    }
}
