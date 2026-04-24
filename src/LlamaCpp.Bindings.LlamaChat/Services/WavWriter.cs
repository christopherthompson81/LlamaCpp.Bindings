using System;
using System.Buffers.Binary;

namespace LlamaCpp.Bindings.LlamaChat.Services;

/// <summary>
/// Minimal RIFF/WAVE writer — 16-bit signed PCM, mono or stereo.
/// We only need it to wrap captured mic samples into a format miniaudio
/// (inside mtmd) can decode, so the 44-byte canonical header is enough.
/// </summary>
internal static class WavWriter
{
    public static byte[] BuildPcm16(ReadOnlySpan<short> samples, int sampleRate, int channels = 1)
    {
        int dataBytes = samples.Length * sizeof(short);
        int fileSize = 36 + dataBytes;
        int byteRate = sampleRate * channels * sizeof(short);
        short blockAlign = (short)(channels * sizeof(short));

        var buffer = new byte[44 + dataBytes];
        var span = buffer.AsSpan();

        // RIFF header
        span[0] = (byte)'R'; span[1] = (byte)'I'; span[2] = (byte)'F'; span[3] = (byte)'F';
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(4, 4), fileSize);
        span[8] = (byte)'W'; span[9] = (byte)'A'; span[10] = (byte)'V'; span[11] = (byte)'E';

        // fmt chunk
        span[12] = (byte)'f'; span[13] = (byte)'m'; span[14] = (byte)'t'; span[15] = (byte)' ';
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(16, 4), 16);        // chunk size
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(20, 2), 1);         // PCM
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(22, 2), (short)channels);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(24, 4), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(28, 4), byteRate);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(32, 2), blockAlign);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(34, 2), 16);        // bits per sample

        // data chunk
        span[36] = (byte)'d'; span[37] = (byte)'a'; span[38] = (byte)'t'; span[39] = (byte)'a';
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(40, 4), dataBytes);

        for (int i = 0; i < samples.Length; i++)
            BinaryPrimitives.WriteInt16LittleEndian(span.Slice(44 + i * 2, 2), samples[i]);

        return buffer;
    }
}
