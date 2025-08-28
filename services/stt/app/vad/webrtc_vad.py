"""
WebRTC VAD tabanlı basit segmentleyici.

Giriş: 16 kHz (veya belirtilen), mono, 16-bit PCM (little-endian) byte dizisi.
Çıkış: "konuşma" olarak sınıflanan segmentlerin ham PCM16 byte blokları.

Notlar:
- WebRTC VAD yalnızca 10ms / 20ms / 30ms kare uzunluklarını kabul eder.
- 'aggressiveness' 0..3 arasıdır; yükseldikçe daha muhafazakâr (daha az konuşma) olur.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Generator, Iterable, List

import webrtcvad


class Frame:
    """Tek bir PCM16 çerçevesi (kare)."""

    __slots__ = ("bytes", "timestamp", "duration")

    def __init__(self, bytes_data: bytes, timestamp: float, duration: float):
        self.bytes = bytes_data        # raw 16-bit PCM (mono, little-endian)
        self.timestamp = timestamp     # saniye cinsinden başlangıç zamanı
        self.duration = duration       # saniye cinsinden kare süresi


def _normalize_frame_ms(frame_ms: int) -> int:
    """VAD'in kabul ettiği en yakın değerlerden birine (10/20/30) yuvarla."""
    if frame_ms <= 10:
        return 10
    if frame_ms <= 20:
        return 20
    return 30


def frame_generator(
    frame_duration_ms: int,
    pcm16: bytes,
    sample_rate: int,
) -> Generator[Frame, None, None]:
    """
    PCM16 byte dizisini, belirtilen kare süresine göre Frame'lere böler.

    frame_duration_ms: 10 / 20 / 30 (diğer değerler en yakınına yuvarlanır)
    """
    frame_ms = _normalize_frame_ms(frame_duration_ms)
    # 16-bit mono: örnek başına 2 byte
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0) * 2)

    offset = 0
    timestamp = 0.0
    duration = bytes_per_frame / (2.0 * sample_rate)  # saniye

    total = len(pcm16)
    while offset + bytes_per_frame <= total:
        chunk = pcm16[offset : offset + bytes_per_frame]
        yield Frame(chunk, timestamp, duration)
        offset += bytes_per_frame
        timestamp += duration


def vad_collector(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    aggressiveness: int,
    frames: Iterable[Frame],
) -> Generator[bytes, None, None]:
    """
    WebRTC VAD ile konuşma segmentleri üretir.

    - padding_duration_ms: konuşma/sessizlik geçişlerinde tampon uzunluğu.
    - aggressiveness: 0..3 (yüksek değer = daha agresif sessizlik kabulü).
    - frames: frame_generator çıktısı.

    Üretilen her eleman, bir konuşma segmentinin (başlangıçtan bitişe) ham PCM16 bytes'ıdır.
    """
    vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
    frame_ms = _normalize_frame_ms(frame_duration_ms)

    num_padding_frames = max(1, int(padding_duration_ms / frame_ms))
    ring_buffer: Deque[tuple[Frame, bool]] = deque(maxlen=num_padding_frames)

    triggered = False
    voiced_frames: List[Frame] = []

    def _emit_segment(frames_list: List[Frame]) -> bytes:
        if not frames_list:
            return b""
        return b"".join(f.bytes for f in frames_list)

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            # Bekleme durumundayız: yeterli sayıda "speech" oranı görülürse tetikle
            ring_buffer.append((frame, is_speech))
            num_voiced = sum(1 for _, s in ring_buffer if s)
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # ring buffer’daki kareleri de segment başlangıcına dahil et
                voiced_frames.extend(f for f, _ in ring_buffer)
                ring_buffer.clear()
        else:
            # Konuşma sırasında: kareleri biriktir
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = sum(1 for _, s in ring_buffer if not s)

            # Yeterli sessizlik görülürse segmenti bitir
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield _emit_segment(voiced_frames)
                voiced_frames.clear()
                ring_buffer.clear()

    # Akış biterse, elde kalan konuşmayı da gönder
    if voiced_frames:
        yield _emit_segment(voiced_frames)
