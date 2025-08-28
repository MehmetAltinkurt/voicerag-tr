from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
import json
import os
import time

from .. import config
from ..engines.vosk_engine import VoskEngine
from ..engines.whispercpp_engine import WhisperCppEngine
from ..vad.webrtc_vad import frame_generator, vad_collector

router = APIRouter(prefix="/stt", tags=["STT-Streaming"])


def _get_engine(engine_name: str):
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
        # Bu iskelette Whisper.cpp yalnızca CLI ile batch çalışır
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


@router.websocket("/stream")
async def stream(
    ws: WebSocket,
    engine: str = Query(default=config.ENGINE_DEFAULT, description="vosk | whispercpp"),
    lang: str = Query(default=config.LANG_DEFAULT),
    sample_rate: int = Query(default=config.SAMPLE_RATE_DEFAULT, ge=8000, le=48000),
    vad_aggr: int = Query(default=config.VAD_AGGRESSIVENESS),
    vad_frame_ms: int = Query(default=config.VAD_FRAME_MS),
    vad_padding_ms: int = Query(default=config.VAD_PADDING_MS),
):
    """
    WebSocket streaming endpoint.

    Beklenen ikili (binary) mesaj formatı:
      - 16 kHz (varsayılan), mono, 16-bit PCM (little-endian) raw chunk'lar.

    Metin (text) mesajları:
      - {"type":"start","session_id":"..."}  -> isteğe bağlı
      - {"type":"stop"}                      -> finalize ve kapat
    """
    await ws.accept()

    # Engine hazırla
    try:
        stt = _get_engine(engine)
    except HTTPException as e:
        await ws.send_text(json.dumps({"type": "error", "message": e.detail}))
        await ws.close()
        return

    # Vosk gerçek zamanlı stream'i destekler; Whisper.cpp bu iskelette batch'tir.
    streaming_supported = (stt.__class__.__name__.lower().startswith("vosk"))

    # Streaming başlat
    try:
        stt.stream_init(sample_rate=sample_rate, lang=lang)
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": f"stream_init failed: {e}"}))
        await ws.close()
        return

    # Whisper.cpp gibi batch motorları için buffer + VAD kullanacağız
    buffer = bytearray()

    try:
        while True:
            msg = await ws.receive()

            # ---- JSON TEXT KONTROLÜ ----
            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    # geçersiz JSON, yok say
                    continue

                msg_type = payload.get("type")

                if msg_type == "start":
                    await ws.send_text(json.dumps({"type": "ack", "at": time.time()}))
                    continue

                if msg_type == "stop":
                    # Vosk: kalanları finalize et
                    if streaming_supported:
                        try:
                            fin = stt.stream_finalize()
                            if fin and fin.get("text"):
                                await ws.send_text(json.dumps({
                                    "type": "final",
                                    "text": fin.get("text", ""),
                                    "info": fin.get("raw"),
                                    "words": fin.get("words"),
                                }))
                        except Exception as e:
                            await ws.send_text(json.dumps({"type": "error", "message": f"finalize failed: {e}"}))

                    # Batch (Whisper.cpp) için: VAD ile segmentle ve her birini transcribe et
                    else:
                        if buffer:
                            try:
                                frames = list(frame_generator(vad_frame_ms, bytes(buffer), sample_rate))
                                for segment in vad_collector(sample_rate, vad_frame_ms, vad_padding_ms, vad_aggr, frames):
                                    res = stt.transcribe(segment, sample_rate, lang)
                                    await ws.send_text(json.dumps({
                                        "type": "final",
                                        "text": res.get("text", ""),
                                        "info": res.get("raw"),
                                        "words": res.get("words"),
                                    }))
                            except Exception as e:
                                await ws.send_text(json.dumps({"type": "error", "message": f"VAD/batch failed: {e}"}))
                            finally:
                                buffer.clear()

                    await ws.send_text(json.dumps({"type": "done"}))
                    await ws.close()
                    break

                # Bilinmeyen text mesaj -> geç
                continue

            # ---- BINARY: RAW PCM16 CHUNK ----
            if "bytes" in msg:
                chunk = msg["bytes"]

                if streaming_supported:
                    # Vosk: gerçek zamanlı partial/final gönder
                    try:
                        out = stt.stream_feed(chunk)
                        if out is not None:
                            if out.get("final"):
                                await ws.send_text(json.dumps({
                                    "type": "final",
                                    "text": out.get("text", ""),
                                    "info": out.get("raw"),
                                    "words": out.get("words"),
                                }))
                            else:
                                # partial
                                await ws.send_text(json.dumps({
                                    "type": "partial",
                                    "text": out.get("text", ""),
                                    "info": out.get("raw"),
                                }))
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "error", "message": f"stream_feed failed: {e}"}))
                else:
                    # Whisper.cpp: batch toplama (stop'ta VAD ile finalize)
                    buffer.extend(chunk)

    except WebSocketDisconnect:
        # istemci bağlantıyı kesti -> sessizce çık
        return
    except Exception as e:
        # beklenmeyen hata
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
