from __future__ import annotations
import base64
import httpx
from fastapi import HTTPException
from core.config import settings
from core.audio import ensure_wav_pcm16_mono_16k

async def _call_stt(wav_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        # Varsayım: STT endpoint'i audio/wav içerik kabul ediyor ve
        # JSON {"text": "..."} veya {"transcript": "..."} dönüyor.
        r = await client.post(settings.STT_URL, content=wav_bytes, headers={"Content-Type": "audio/wav"})
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"STT error: {r.text}")
        try:
            data = r.json()
        except Exception:
            raise HTTPException(500, "STT returned non-JSON response")
        text = data.get("text") or data.get("transcript") or data.get("result")
        if not text:
            raise HTTPException(500, f"STT JSON missing text: {data}")
        return text

async def _call_tts(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        # Varsayım: TTS endpoint'i JSON {"text": "..."} alıyor.
        # audio/wav byte döndürebilir ya da JSON içinde base64 olabilir.
        r = await client.post(settings.TTS_URL, json={"text": text})
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"TTS error: {r.text}")

        ctype = r.headers.get("content-type", "")
        if ctype.startswith("audio/"):
            return r.content
        # İçerik türü yoksa yine doğrudan bytes olabilir
        try:
            data = r.json()
        except Exception:
            return r.content
        # Ortak JSON şekilleri
        if isinstance(data, dict):
            if "audio" in data and isinstance(data["audio"], dict) and "data" in data["audio"]:
                return base64.b64decode(data["audio"]["data"]) # {audio:{data:<b64>}}
            if "audio_base64" in data:
                return base64.b64decode(data["audio_base64"]) # {audio_base64:<b64>}
            if "wav_base64" in data:
                return base64.b64decode(data["wav_base64"]) # {wav_base64:<b64>}
        raise HTTPException(500, f"Unrecognized TTS response shape: {data}")




async def run_echo_pipeline(input_wav: bytes) -> tuple[str, bytes]:
    """Echo pipeline: Normalize → STT → TTS.
    Dönüş: (transcript, wav_bytes)
    """
    wav16k = ensure_wav_pcm16_mono_16k(input_wav)
    text = await _call_stt(wav16k)
    out_wav = await _call_tts(text)
    return text, out_wav