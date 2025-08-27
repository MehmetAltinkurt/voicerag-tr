# stt_api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydub import AudioSegment
import json, os, subprocess, tempfile
from utils_audio import to_wav_mono_16k

# --- VOSK Kurulumu ---
from vosk import Model, KaldiRecognizer

# Vosk model klasörü (indirip buraya koy)
VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "models/vosk-tr")  # örn: models/vosk-model-small-tr
VOSK_SR = 16000

# Whisper.cpp yolları
WHISPER_BIN = os.getenv("WHISPER_BIN", "./build/bin/whisper-cli")  # Windows'ta .exe yolu
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "./models/ggml-base.bin")

app = FastAPI(title="Free STT: Vosk + whisper.cpp")

# Vosk modeli bir kez yüklenir (hızlı başlasın diye opsiyonel: lazy load da yapabilirsin)
_vosk_model = None
def get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        if not os.path.isdir(VOSK_MODEL_DIR):
            raise RuntimeError(f"Vosk modeli bulunamadı: {VOSK_MODEL_DIR}")
        _vosk_model = Model(VOSK_MODEL_DIR)
    return _vosk_model

@app.post("/stt")
async def transcribe(
    file: UploadFile = File(...),
    provider: str = Form("vosk"),        # "vosk" | "whispercpp"
    language: str = Form("tr")           # whisper.cpp’de -l tr kullanırız
):
    # Yüklenen dosyayı geçici diske yaz
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        raw_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        # Her iki motor için de WAV 16k mono hazırla
        wav_path = to_wav_mono_16k(raw_path)

        if provider.lower() == "vosk":
            model = get_vosk_model()
            rec = KaldiRecognizer(model, VOSK_SR)
            # Parça parça beslemek istersen pydub ile:
            audio = AudioSegment.from_file(wav_path)
            chunk_ms = 8000
            for i in range(0, len(audio), chunk_ms):
                chunk = audio[i:i+chunk_ms].raw_data
                rec.AcceptWaveform(chunk)

            # Sonuçları toparla
            final = json.loads(rec.FinalResult())
            text = final.get("text", "").strip()
            return {"engine": "vosk", "lang": "tr-TR", "text": text}

        elif provider.lower() == "whispercpp":
            # whisper.cpp CLI ile çağır
            if not (os.path.isfile(WHISPER_BIN) and os.path.isfile(WHISPER_MODEL)):
                raise HTTPException(500, "whisper.cpp binary veya model bulunamadı.")
            cmd = [
                WHISPER_BIN,
                "-m", WHISPER_MODEL,
                "-l", language,
                "-t", str(os.cpu_count() or 4),
                "-otxt",     # çıktı .txt dosyası
                "-f", wav_path
            ]
            # Çıktı aynı klasöre <wav>.txt olarak yazılır
            subprocess.run(cmd, check=True)
            txt_path = wav_path + ".txt"
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            return {"engine": "whisper.cpp", "lang": language, "text": text}

        else:
            raise HTTPException(400, "Geçersiz provider. 'vosk' veya 'whispercpp' kullanın.")

    finally:
        # Geçici dosyaları temizle
        for p in [raw_path, wav_path if 'wav_path' in locals() else None, (wav_path + ".txt") if 'wav_path' in locals() and os.path.exists(wav_path + ".txt") else None]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
