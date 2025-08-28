"""
Audio yardımcıları:
- Her türlü (WAV/FLAC/OGG/çoğu MP3*) ses verisini okuyup float32 mono'ya çevirir.
- Hedef örnekleme hızına (default 16 kHz) resample eder.
- 16-bit PCM (mono) byte dizisine dönüştürür.

*Not: MP3 desteği ortamınızdaki libsndfile kurulumuna bağlıdır.
"""

from typing import Tuple
import io

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    """
    Çok kanallı sesleri (N, C) -> (N,) mono'ya indirger, float32 döner.
    """
    if data.ndim == 1:
        # (N,)
        mono = data.astype(np.float32, copy=False)
    else:
        # (N, C) -> kanal ortalaması
        mono = data.mean(axis=1).astype(np.float32, copy=False)
    return mono


def decode_audio_to_float_mono(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Bytes içeriğini (container fark etmeksizin) float32 mono dizi + örnekleme hızı olarak döndür.
    """
    # soundfile doğrudan bytes okuyabilir
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    mono = _to_mono_float32(data)
    return mono, int(sr)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    resample_poly ile yüksek kaliteli yeniden örnekleme.
    """
    if sr_in == sr_out:
        return x
    # r = sr_out / sr_in oranı -> resample_poly(up=sr_out, down=sr_in)
    y = resample_poly(x, sr_out, sr_in)
    return y.astype(np.float32, copy=False)


def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    """
    [-1, 1] aralığını 16-bit tamsayıya çevirip bytes döndürür.
    """
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def bytes_to_pcm16(audio_bytes: bytes, target_sr: int = 16000) -> bytes:
    """
    Yüklenen ses dosyasını:
      1) decode -> float32 mono
      2) target_sr'e resample
      3) int16 PCM bytes
    şeklinde dönüştürür.
    """
    x, sr = decode_audio_to_float_mono(audio_bytes)
    x_rs = resample_to(x, sr_in=sr, sr_out=target_sr)
    return float_to_pcm16_bytes(x_rs)
