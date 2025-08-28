import json
from typing import Dict, Any, Optional

from vosk import Model, KaldiRecognizer

from .base import STTEngine

# ------------------------------------------------------------
# Model cache:
#  - Her istekte modeli tekrar okumamak için aynı model yolu
#    üzerinden tek bir Model örneğini paylaşırız.
# ------------------------------------------------------------
_MODEL_CACHE: Dict[str, Model] = {}


def _get_model(model_path: str) -> Model:
    m = _MODEL_CACHE.get(model_path)
    if m is None:
        m = Model(model_path)
        _MODEL_CACHE[model_path] = m
    return m


class VoskEngine(STTEngine):
    """
    Vosk tabanlı STT motoru.

    Özellikler:
    - Batch transkripsiyon: PCM16 (mono, 16-bit) + örnekleme hızı (sr) ile tam metin.
    - Streaming destekleri: partial / final sonuçlar (KaldiRecognizer.PartialResult/Result).

    NOT: Bu sınıf "ham" PCM16 bekler. Router tarafında yüklenen dosya zaten
    utils.audio.bytes_to_pcm16 ile 16kHz mono 16-bit PCM'e dönüştürülüyor.
    """
    name = "vosk"

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        use_words: bool = True,
        grammar: Optional[str] = None,
    ):
        """
        :param model_path: Vosk model klasörü (ör: .../vosk-model-small-tr-0.4)
        :param sample_rate: Varsayılan örnekleme hızı (Hz)
        :param use_words: Kelime zamanlamalarını (words) üret
        :param grammar: (Opsiyonel) JSON string olarak grammar (örn: '["merhaba", "günaydın"]')
                        Verilirse tanıma bu kelimelerle sınırlandırılır.
        """
        self.model = _get_model(model_path)
        self.default_sr = sample_rate
        self.use_words = use_words
        self.grammar = grammar

        self._recognizer: Optional[KaldiRecognizer] = None  # streaming için

    # ----------------------------
    # Yardımcılar
    # ----------------------------
    def _make_recognizer(self, sr: int) -> KaldiRecognizer:
        rec = KaldiRecognizer(self.model, sr)
        if self.use_words:
            rec.SetWords(True)
        if self.grammar:
            # Grammar JSON string bekler. Örn: '["merhaba", "günaydın"]'
            rec.SetGrammar(self.grammar)
        return rec

    @staticmethod
    def _parse_final_result(res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vosk FinalResult/Result çıktısını normalize eder.
        Beklenen alanlar:
          - text: tüm metin
          - result: kelime listesi (start/end/conf/word)
        """
        text = res.get("text", "") or ""
        words = res.get("result", None)  # yoksa None olarak bırak
        return {
            "text": text,
            "words": words,  # kelime zamanlamaları (opsiyonel)
            "raw": res,      # ham vosk çıktısı
        }

    @staticmethod
    def _parse_partial_result(res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vosk PartialResult çıktısı:
          - partial: geçici/henüz finalize olmamış metin
        """
        partial = res.get("partial", "") or ""
        return {
            "text": partial,
            "raw": res,
        }

    # ----------------------------
    # Batch
    # ----------------------------
    def transcribe(self, pcm16: bytes, sample_rate: int, lang: str = "tr") -> Dict[str, Any]:
        """
        Tüm audioyu tek seferde tanır ve final sonucu döndürür.
        """
        rec = self._make_recognizer(sample_rate)
        # Tek parça besleme (küçük/orta boy dosyalar için yeterli)
        rec.AcceptWaveform(pcm16)
        res = json.loads(rec.FinalResult())
        return self._parse_final_result(res)

    # ----------------------------
    # Streaming
    # ----------------------------
    def stream_init(self, sample_rate: int, lang: str = "tr") -> None:
        """
        Streaming oturumu başlatır (recognizer hazırlar).
        """
        self._recognizer = self._make_recognizer(sample_rate)

    def stream_feed(self, pcm16: bytes) -> Optional[Dict[str, Any]]:
        """
        Streaming sırasında chunk besle.
        - Vosk, AcceptWaveform True dönerse bir segmenti finalize etmiş demektir.
        - False ise PartialResult döner (geçici metin).
        """
        if self._recognizer is None:
            return None

        ok = self._recognizer.AcceptWaveform(pcm16)
        if ok:
            # finalize edilen segment
            res = json.loads(self._recognizer.Result())
            parsed = self._parse_final_result(res)
            return {"final": True, **parsed}
        else:
            # geçici/partial metin
            part = json.loads(self._recognizer.PartialResult())
            parsed = self._parse_partial_result(part)
            return {"final": False, **parsed}

    def stream_finalize(self) -> Optional[Dict[str, Any]]:
        """
        Streaming bittiğinde çağır. Kalan buffer varsa finalize eder.
        """
        if self._recognizer is None:
            return None
        res = json.loads(self._recognizer.FinalResult())
        parsed = self._parse_final_result(res)
        return {"final": True, **parsed}
