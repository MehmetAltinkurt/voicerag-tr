import os

# .env otomatik yükleme (python-dotenv yüklüyse)
try:
    from dotenv import load_dotenv  # requirements'ta yoksa sorun değil
    load_dotenv()
except Exception:
    pass

# Bu dosyanın bulunduğu yerden proje köküne göre relatif path çözmek için
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _abs_path(p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(BASE_DIR, p))

# ===========================
# STT GENEL AYARLAR
# ===========================
ENGINE_DEFAULT = os.getenv("ENGINE_DEFAULT", "vosk")
LANG_DEFAULT = os.getenv("LANG_DEFAULT", "tr")
SAMPLE_RATE_DEFAULT = int(os.getenv("SAMPLE_RATE_DEFAULT", "16000"))

# VAD (WebRTC)
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0..3
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))             # 10/20/30 ms
VAD_PADDING_MS = int(os.getenv("VAD_PADDING_MS", "600"))        # ms

# ===========================
# VOSK
# ===========================
# Örn: ./models/vosk-model-small-tr-0.4
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./models/vosk-model-small-tr-0.3")
VOSK_MODEL_PATH = _abs_path(VOSK_MODEL_PATH)

# ===========================
# WHISPER.CPP
# ===========================
# HTTP server (henüz kullanılmıyor; ileride desteklenecek)
WHISPERCPP_SERVER_URL = os.getenv("WHISPERCPP_SERVER_URL", "").strip()

# CLI binary ve model dosyası (Windows örneği: whisper-cli.exe)
WHISPERCPP_CLI_PATH = os.getenv("WHISPERCPP_CLI_PATH", "./bin/whisper-cli.exe")
WHISPERCPP_MODEL_PATH = os.getenv("WHISPERCPP_MODEL_PATH", "./bin/ggml-base.bin")
WHISPERCPP_THREADS = int(os.getenv("WHISPERCPP_THREADS", "4"))

WHISPERCPP_CLI_PATH = _abs_path(WHISPERCPP_CLI_PATH)
WHISPERCPP_MODEL_PATH = _abs_path(WHISPERCPP_MODEL_PATH)
