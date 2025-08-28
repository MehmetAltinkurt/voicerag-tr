from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class STTEngine(ABC):
    """Tüm STT motorları için ortak arayüz."""

    # Motor adı (örn. "vosk", "whispercpp")
    name: str = "base"

    @abstractmethod
    def transcribe(self, pcm16: bytes, sample_rate: int, lang: str = "tr") -> Dict[str, Any]:
        """
        Batch transkripsiyon: tamamlanmış ses verisinden metin döndürür.
        Beklenen dönüş:
          {
            "text": "...",
            # opsiyonel:
            "words": [...],     # kelime zamanlamaları
            "segments": [...],  # segment listesi
            "raw": {...}        # motorun ham çıktısı
          }
        """
        raise NotImplementedError

    # ---- Streaming API (opsiyonel) ----
    def stream_init(self, sample_rate: int, lang: str = "tr") -> None:
        """Streaming için motoru başlat (gerekirse)."""
        pass

    def stream_feed(self, pcm16: bytes) -> Optional[Dict[str, Any]]:
        """
        Streaming sırasında her chunk için çağrılır.
        Dönerse örnek bir sözlük:
          {"final": False, "text": "...", "raw": {...}}
        veya finalize olduğunda:
          {"final": True, "text": "...", "words": [...], "raw": {...}}
        """
        return None

    def stream_finalize(self) -> Optional[Dict[str, Any]]:
        """Streaming bittiğinde kalan buffer'ı finalize et."""
        return None
