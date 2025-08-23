# services/tts/app/main.py
import os
import tempfile
import subprocess
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from prometheus_fastapi_instrumentator import Instrumentator


class Settings(BaseSettings):
    SERVICE_NAME: str = "tts"
    VERSION: str = "0.1.0"
    PORT: int = 8002

    # Piper
    PIPER_BIN: str
    VOICE_MODEL: str
    VOICE_CONFIG: Optional[str] = None  # bazı paketlerde gerekmez

    # Ses/Prosodi
    SAMPLE_RATE: int = 22050
    DEFAULT_LENGTH_SCALE: float = 1.0
    DEFAULT_NOISE_SCALE: float = 0.667
    DEFAULT_NOISE_W: float = 0.8

    class Config:
        env_file = ".env"


settings = Settings()
app = FastAPI(title="VoiceRAGTR - TTS Service", version=settings.VERSION)

READY = False


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w: Optional[float] = None
    sample_rate: Optional[int] = None  # override etmek istersen


@app.on_event("startup")
def on_startup():
    global READY
    # Piper binary ve model var mı?
    for path in [settings.PIPER_BIN, settings.VOICE_MODEL]:
        if not os.path.isfile(path):
            raise RuntimeError(f"Gerekli dosya bulunamadı: {path}")
    READY = True
    Instrumentator().instrument(app).expose(app)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.SERVICE_NAME}


@app.get("/readyz")
def readyz():
    return {"ready": READY}


@app.get("/version")
def version():
    return {"service": settings.SERVICE_NAME, "version": settings.VERSION}


@app.get("/voice")
def voice_info():
    return {
        "model": settings.VOICE_MODEL,
        "config": settings.VOICE_CONFIG,
        "sample_rate": settings.SAMPLE_RATE,
        "defaults": {
            "length_scale": settings.DEFAULT_LENGTH_SCALE,
            "noise_scale": settings.DEFAULT_NOISE_SCALE,
            "noise_w": settings.DEFAULT_NOISE_W,
        },
    }


@app.post("/speak", responses={200: {"content": {"audio/wav": {}}}})
def speak(req: SpeakRequest):
    """
    Piper CLI'yi çağırıp WAV döndürür.
    - text → stdin
    - WAV → geçici dosya → response
    """
    if not READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Boş metin")

    # Parametreleri topla (override varsa kullan)
    length_scale = req.length_scale or settings.DEFAULT_LENGTH_SCALE
    noise_scale = req.noise_scale or settings.DEFAULT_NOISE_SCALE
    noise_w = req.noise_w or settings.DEFAULT_NOISE_W
    sample_rate = req.sample_rate or settings.SAMPLE_RATE

    # Piper komutu
    # Not: Piper tipik olarak stdin'den metin alır ve -f ile dosyaya yazar.
    cmd = [
        settings.PIPER_BIN,
        "-m", settings.VOICE_MODEL,
        "-f", "-",  # stdout (bazı build'lerde sorun olabilir, o zaman tmp dosyaya yazarız)
        "--length_scale", str(length_scale),
        "--noise_scale", str(noise_scale),
        "--noise_w", str(noise_w),
        "-s", str(sample_rate),
    ]
    if settings.VOICE_CONFIG:
        cmd.extend(["-c", settings.VOICE_CONFIG])

    # Bazı Piper binary sürümleri stdout'a güvenilir yazmayabiliyor.
    # Bu durumda tmp dosya yaklaşımına düşelim.
    try_stdout = True

    if try_stdout:
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
            headers = {"Content-Disposition": 'inline; filename="speech.wav"'}
            return JSONResponse(content={}, headers=headers, media_type="audio/wav", status_code=200)  # placeholder
        except Exception:
            # stdout yöntemi başarısızsa tmp dosya yaklaşımına geç
            pass

    # ---- TMP dosya yaklaşımı ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    # stdout yerine dosyaya yaz
    cmd_file = cmd.copy()
    # stdout'tan "-" kaldır, dosyaya yazalım
    idx = cmd_file.index("-f")
    cmd_file[idx + 1] = tmp_path

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
        raise HTTPException(status_code=500, detail=f"Piper hata: {err or str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Piper çalıştırılamadı: {e}")

    # WAV dosyasını döndür
    return FileResponse(
        tmp_path,
        media_type="audio/wav",
        filename="speech.wav",
        headers={"Cache-Control": "no-store"},
        background=None,  # istersen background task ile sil
    )
