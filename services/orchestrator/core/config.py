# services/orchestrator/core/config.py
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class Settings:
    STT_URL: str = os.getenv("VR_STT_HTTP_URL", "http://localhost:8003/transcribe")
    TTS_URL: str = os.getenv("VR_TTS_HTTP_URL", "http://localhost:8002/speak")
    SAMPLE_RATE: int = int(os.getenv("VR_SAMPLE_RATE", 16000))

settings = Settings()