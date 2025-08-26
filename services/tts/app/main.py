# services/tts/app/main.py
import os
import json
import tempfile
import subprocess
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from prometheus_fastapi_instrumentator import Instrumentator


class Settings(BaseSettings):
    SERVICE_NAME: str = "tts"
    VERSION: str = "0.2.0"
    PORT: int = 8002

    # Backend seçimi: "piper" (yerel) veya "gtts" (bulut/online)
    TTS_BACKEND: Literal["piper", "gtts"] = "piper"

    # Piper
    PIPER_BIN: Optional[str] = None
    VOICE_MODEL: Optional[str] = None
    VOICE_CONFIG: Optional[str] = None  # bazı paketlerde gerekmez

    # Piper prosodi
    SAMPLE_RATE: Optional[int] = None   # None -> modelin doğal değeri
    DEFAULT_LENGTH_SCALE: float = 1.0
    DEFAULT_NOISE_SCALE: float = 0.30   # temiz varsayılan
    DEFAULT_NOISE_W: float = 0.50

    # gTTS (Google Translate TTS; internet gerekir, MP3 üretir)
    GTTS_LANG: str = "tr"
    GTTS_TLD: str = "com"     # com/com.tr vb.
    GTTS_SLOW: bool = False   # yavaş okuma

    class Config:
        env_file = ".env.example"


settings = Settings()
app = FastAPI(title="VoiceRAGTR - TTS Service", version=settings.VERSION)

# middleware/metrics: modül seviyesinde ekleyin
Instrumentator().instrument(app).expose(app)

READY = False

# gTTS import’u opsiyonel (yalnızca gerekince)
def _gtts_synthesize(text: str, lang: str, tld: str, slow: bool, outfile: str) -> None:
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    tts.save(outfile)  # MP3 olarak yazar


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    # Piper parametreleri (gTTS bunları dikkate almaz)
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w: Optional[float] = None
    sample_rate: Optional[int] = None  # sadece Piper


@app.on_event("startup")
def on_startup():
    """Gerekli dosyaları doğrula; servis hazır bayrağını kaldır."""
    global READY
    if settings.TTS_BACKEND == "piper":
        for key, path in {"PIPER_BIN": settings.PIPER_BIN, "VOICE_MODEL": settings.VOICE_MODEL}.items():
            if not path or not os.path.isfile(path):
                raise RuntimeError(f"{key} bulunamadı: {path!r}. Piper için .env'i doldurun.")
    else:
        # gTTS için yerel dosya zorunluluğu yok; internet gerekir.
        pass
    READY = True


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.SERVICE_NAME}


@app.get("/readyz")
def readyz():
    return {"ready": READY, "backend": settings.TTS_BACKEND}


@app.get("/version")
def version():
    return {"service": settings.SERVICE_NAME, "version": settings.VERSION}


@app.get("/voice")
def voice_info():
    return {
        "backend": settings.TTS_BACKEND,
        "piper": {
            "model": settings.VOICE_MODEL,
            "config": settings.VOICE_CONFIG,
            "sample_rate": settings.SAMPLE_RATE,
            "defaults": {
                "length_scale": settings.DEFAULT_LENGTH_SCALE,
                "noise_scale": settings.DEFAULT_NOISE_SCALE,
                "noise_w": settings.DEFAULT_NOISE_W,
            },
        },
        "gtts": {
            "lang": settings.GTTS_LANG,
            "tld": settings.GTTS_TLD,
            "slow": settings.GTTS_SLOW,
        },
    }


@app.post(
    "/speak",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
            }
        }
    },
)
def speak(
    req: SpeakRequest,
    background: BackgroundTasks,
    engine: Optional[Literal["piper", "gtts"]] = Query(
        None, description="Bu çağrı için motoru override et (piper|gtts)"
    ),
):
    """
    TTS üretir. Varsayılan backend .env'den gelir; istersen ?engine=gtts ile override edebilirsin.
    Piper: WAV, gTTS: MP3 döner.
    """
    if not READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Boş metin")

    backend = engine or settings.TTS_BACKEND

    if backend == "gtts":
        # ---- gTTS (MP3) ----
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            mp3_path = tmp.name
        try:
            _gtts_synthesize(text, settings.GTTS_LANG, settings.GTTS_TLD, settings.GTTS_SLOW, mp3_path)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"gTTS başarısız (internet gerekiyor olabilir): {e}"
            )

        def _cleanup(path: str):
            try: os.remove(path)
            except Exception: pass

        background.add_task(_cleanup, mp3_path)
        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            headers={"Cache-Control": "no-store"},
        )

    # ---- Piper (WAV) ----
    length_scale = req.length_scale or settings.DEFAULT_LENGTH_SCALE
    noise_scale = req.noise_scale or settings.DEFAULT_NOISE_SCALE
    noise_w = req.noise_w or settings.DEFAULT_NOISE_W
    sample_rate = req.sample_rate or settings.SAMPLE_RATE or None

    cmd = [
        settings.PIPER_BIN,
        "-m", settings.VOICE_MODEL,
        "-f", "-",  # önce stdout'u deneriz
        "--length_scale", str(length_scale),
        "--noise_scale", str(noise_scale),
        "--noise_w", str(noise_w),
    ]
    if settings.VOICE_CONFIG:
        cmd += ["-c", settings.VOICE_CONFIG]
    if sample_rate:
        cmd += ["-s", str(sample_rate)]

    # 1) stdout hızlı yolu
    try:
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        wav_bytes = proc.stdout
        if not wav_bytes:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
        )
    except Exception:
        pass  # 2) tmp dosya fallback

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    cmd_file = cmd.copy()
    idx = cmd_file.index("-f")
    cmd_file[idx + 1] = wav_path  # stdout yerine dosyaya yaz

    try:
        proc = subprocess.run(
            cmd_file,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        # 3221225781 -> DLL eksikliği ipucu (Windows)
        if e.returncode == 3221225781:
            hint = ("Piper DLL eksikliği (onnxruntime*.dll veya VC++ redist). "
                    "Piper klasörünü ve Visual C++ 2015–2022 x64 kurulumunu kontrol edin.")
            raise HTTPException(status_code=500, detail=f"Piper hata: {hint}")
        raise HTTPException(status_code=500, detail=f"Piper hata: {err or str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Piper çalıştırılamadı: {e}")

    def _cleanup(path: str):
        try: os.remove(path)
        except Exception: pass

    background.add_task(_cleanup, wav_path)
    return FileResponse(
        wav_path,
        media_type="audio/wav",
        filename="speech.wav",
        headers={"Cache-Control": "no-store"},
    )
