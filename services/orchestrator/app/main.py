from __future__ import annotations
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipelines.echo import run_echo_pipeline
from core.config import settings

app = FastAPI(title="VoiceRAG Orchestrator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"ok": True, "stt": settings.STT_URL, "tts": settings.TTS_URL}

@app.post("/echo")
async def echo(file: UploadFile = File(...)):
    """
    WAV al → STT → TTS → transcript + base64 wav döndür
    """
    try:
        raw = await file.read()
        text, wav_bytes = await run_echo_pipeline(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"echo pipeline error: {e}")

    return {
        "text": text,
        "audio": {
            "format": "wav",
            "sample_rate": settings.SAMPLE_RATE,
            "data": base64.b64encode(wav_bytes).decode("ascii"),
        },
    }
