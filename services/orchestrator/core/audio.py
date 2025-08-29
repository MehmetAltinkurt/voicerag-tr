# services/orchestrator/core/audio.py
from __future__ import annotations
import io
import wave
import audioop
from .config import settings




def ensure_wav_pcm16_mono_16k(wav_bytes: bytes) -> bytes:
    """Arbitrary WAV → PCM16 mono 16kHz. Yalnızca stdlib (wave+audioop)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sampwidth != 2:
        frames = audioop.lin2lin(frames, sampwidth, 2)
        sampwidth = 2

    if n_channels != 1:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
        n_channels = 1

    if framerate != settings.SAMPLE_RATE:
        frames, _ = audioop.ratecv(frames, 2, 1, framerate, settings.SAMPLE_RATE, None)
        framerate = settings.SAMPLE_RATE

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(settings.SAMPLE_RATE)
        out.writeframes(frames)
    return out_buf.getvalue()