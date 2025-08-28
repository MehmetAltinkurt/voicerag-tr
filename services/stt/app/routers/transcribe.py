from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from typing import Optional
import os

from ..models.schemas import TranscribeResponse
from ..utils.audio import bytes_to_pcm16
from ..engines.vosk_engine import VoskEngine
from ..engines.whispercpp_engine import WhisperCppEngine
from .. import config

router = APIRouter(prefix="/stt", tags=["STT"])


def _get_engine(engine_name: str):
    """
    İstenen motoru hazırla ve döndür.
    Ortam/ayar doğrulamaları burada yapılır.
    """
    name = (engine_name or "").lower()

    if name == "vosk":
        if not os.path.isdir(config.VOSK_MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Vosk model path not found: {config.VOSK_MODEL_PATH}",
            )
        return VoskEngine(
            model_path=config.VOSK_MODEL_PATH,
            sample_rate=config.SAMPLE_RATE_DEFAULT,
            use_words=True,
        )

    if name == "whispercpp":
        # Bu iskelette Whisper.cpp yalnızca CLI ile çağrılıyor.
        if not os.path.isfile(config.WHISPERCPP_CLI_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"whisper-cli not found: {config.WHISPERCPP_CLI_PATH}",
            )
        if not os.path.isfile(config.WHISPERCPP_MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Whisper model not found: {config.WHISPERCPP_MODEL_PATH}",
            )
        return WhisperCppEngine(
            cli_path=config.WHISPERCPP_CLI_PATH,
            model_path=config.WHISPERCPP_MODEL_PATH,
            threads=config.WHISPERCPP_THREADS,
            server_url=config.WHISPERCPP_SERVER_URL,
        )

    raise HTTPException(status_code=400, detail=f"Unknown engine: {engine_name}")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    engine: str = Query(default=config.ENGINE_DEFAULT, description="vosk | whispercpp"),
    lang: str = Query(default=config.LANG_DEFAULT),
    sample_rate: int = Query(default=config.SAMPLE_RATE_DEFAULT, ge=8000, le=48000),
):
    """
    Batch transkripsiyon: dosya yükle → metin al.

    Örnek:
    curl -X POST "http://localhost:8001/stt/transcribe?engine=vosk&lang=tr" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@sample.wav"
    """
    data = await file.read()

    # Yüklenen ses dosyasını 16-bit PCM (mono) ve istenen örnekleme hızına çevir
    try:
        pcm16 = bytes_to_pcm16(data, target_sr=sample_rate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode/resample failed: {e}")

    stt = _get_engine(engine)

    try:
        result = stt.transcribe(pcm16, sample_rate=sample_rate, lang=lang)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")

    text = result.get("text", "")
    words = result.get("words")
    segments = result.get("segments")  # bazı motorlarda olabilir
    raw = result.get("raw")

    return TranscribeResponse(
        engine=stt.name,
        lang=lang,
        text=text,
        words=words,
        segments=segments,
        info=raw,
    )