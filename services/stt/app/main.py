from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.transcribe import router as transcribe_router
from .routers.stream import router as stream_router

app = FastAPI(
    title="VoiceRAG STT Service",
    version="0.1.0",
    description="Vosk ve Whisper.cpp tabanlı modüler STT servisi",
)

# Gerekirse kökenleri sınırlandırabilirsin (şimdilik açık)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routerlar
app.include_router(transcribe_router)
app.include_router(stream_router)

@app.get("/health")
def health():
    return {"ok": True, "service": "stt", "version": "0.1.0"}
