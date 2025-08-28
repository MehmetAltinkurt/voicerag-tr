import os
import json
import tempfile
import wave
import subprocess
from typing import Dict, Any, Optional, List

from .base import STTEngine


class WhisperCppEngine(STTEngine):
    """
    whisper.cpp entegrasyonu (CLI üzerinden).

    Notlar:
    - PCM16 (mono, 16-bit) bytes alır; geçici bir WAV dosyasına yazıp CLI'ı çağırır.
    - JSON çıktısı için '-oj' ve metin çıktısı için '-otxt' kullanır.
    - Önce JSON'u (varsa) okur; yoksa TXT'ye düşer; ikisi de yoksa stdout'u dener.
    """
    name = "whispercpp"

    def __init__(
        self,
        server_url: str = "",
        cli_path: str = "",
        model_path: str = "",
        threads: int = 4,
    ):
        self.server_url = (server_url or "").strip()   # şu an kullanılmıyor
        self.cli_path = (cli_path or "").strip()
        self.model_path = (model_path or "").strip()
        self.threads = int(threads) if threads else 4

        if not self.cli_path:
            raise RuntimeError("WHISPERCPP_CLI_PATH belirtilmeli.")
        if not os.path.isfile(self.cli_path):
            raise RuntimeError(f"whisper-cli bulunamadı: {self.cli_path}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise RuntimeError(f"Whisper model dosyası bulunamadı: {self.model_path}")

    # ------------- Yardımcılar -------------

    @staticmethod
    def _write_temp_wav(pcm16: bytes, sample_rate: int) -> str:
        """
        PCM16 mono veriyi geçici bir WAV dosyasına yazar, yolunu döndürür.
        """
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # wave modülüyle açacağız
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16)
        except Exception:
            # yazma başarısızsa dosyayı temizle
            try:
                os.unlink(wav_path)
            except Exception:
                pass
            raise
        return wav_path

    @staticmethod
    def _safe_read(path: str) -> Optional[str]:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception:
            return None
        return None

    @staticmethod
    def _parse_json_result(json_text: str) -> Dict[str, Any]:
        """
        whisper.cpp '-oj' çıktısını olabildiğince esnek biçimde yorumlar.
        Beklenen alanlar implementasyona göre değişebilir; metni ve segmentleri çıkarmaya çalışır.
        """
        try:
            obj = json.loads(json_text)
        except Exception:
            return {}

        text = ""
        segments_out: List[Dict[str, Any]] = []

        # Bazı sürümlerde doğrudan 'text' alanı olabilir
        if isinstance(obj, dict):
            text = obj.get("text", "") or ""

            # Segmentler varsa normalize edelim
            segs = obj.get("segments") or obj.get("result") or []
            if isinstance(segs, list):
                for s in segs:
                    if not isinstance(s, dict):
                        continue
                    segments_out.append({
                        "start": s.get("start"),
                        "end": s.get("end"),
                        "text": s.get("text", ""),
                        "avg_logprob": s.get("avg_logprob"),
                        "compression_ratio": s.get("compression_ratio"),
                        "no_speech_prob": s.get("no_speech_prob"),
                    })

            # Eğer text boşsa segmentlerin text'lerini birleştir
            if not text and segments_out:
                text = " ".join((s.get("text") or "").strip() for s in segments_out).strip()

        return {"text": text, "segments": segments_out, "raw": obj}

    # ------------- Ana API -------------

    def transcribe(self, pcm16: bytes, sample_rate: int, lang: str = "tr") -> Dict[str, Any]:
        """
        PCM16 (mono, 16-bit) + örnekleme hızı alır; whisper.cpp CLI ile transkripsiyon yapar.
        Öncelik: JSON (-oj) -> TXT (-otxt) -> stdout.
        """
        wav_path = self._write_temp_wav(pcm16, sample_rate)
        base_prefix = wav_path[:-4]  # .wav'ı kaldır
        out_prefix = base_prefix  # '-of' için prefix

        json_path = f"{out_prefix}.json"
        txt_path = f"{out_prefix}.txt"

        # Komut argümanları
        # Not: '-l {lang}' ile dili belirtiyoruz (örn. 'tr'); otomatik algı istersen 'auto'
        args = [
            self.cli_path,
            "-m", self.model_path,
            "-f", wav_path,
            "-t", str(self.threads),
            "-l", lang or "auto",
            "-of", out_prefix,  # çıktı dosyaları için prefix
            "-oj",              # JSON çıktı
            "-otxt",            # TXT çıktı
        ]

        # Çalıştır
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,     # hatayı kendimiz ele alacağız
                shell=False,
            )
        except Exception as e:
            # WAV dosyasını temizle ve patlat
            try:
                os.unlink(wav_path)
            except Exception:
                pass
            raise RuntimeError(f"whisper.cpp çalıştırma hatası: {e}")

        # Çıktıları okumaya çalış
        text = ""
        segments: Optional[List[Dict[str, Any]]] = None
        raw_info: Dict[str, Any] = {"source": "whisper.cpp", "returncode": proc.returncode}

        # Önce JSON
        json_text = self._safe_read(json_path)
        if json_text:
            parsed = self._parse_json_result(json_text)
            text = parsed.get("text", "") or text
            segments = parsed.get("segments")

        # Sonra TXT (JSON boşsa)
        if not text:
            txt_text = self._safe_read(txt_path)
            if txt_text:
                text = txt_text.strip()

        # En sonda stdout fallback
        if not text and proc.stdout:
            text = proc.stdout.strip()

        # Hata kontrolü
        if proc.returncode != 0 and not text:
            err = (proc.stderr or "").strip()
            # Temizlik yapmadan önce hata mesajını hazırla
            msg = f"whisper.cpp CLI hata (code {proc.returncode}): {err}"
            # temizle ve yükselt
            self._cleanup_paths([wav_path, json_path, txt_path])
            raise RuntimeError(msg)

        # Temizlik
        self._cleanup_paths([wav_path, json_path, txt_path])

        return {
            "text": text or "",
            "segments": segments,
            "raw": raw_info,
        }

    @staticmethod
    def _cleanup_paths(paths: List[str]) -> None:
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass
