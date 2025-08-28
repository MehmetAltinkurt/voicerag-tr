# STT Service (Vosk + Whisper.cpp)

- Batch: `POST /stt/transcribe`
- Streaming: `WS /stt/stream`
- Engine: `engine=vosk | whispercpp`
- Dil: `lang=tr` (varsayılan)

## Kurulum
```bash
cd services/stt
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
