import os
import base64
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

DEFAULT_ORCH_URL = os.getenv("ORCH_ECHO_URL", "http://localhost:8010/echo")

st.set_page_config(page_title="Echo (STT→TTS)", layout="centered")
st.title("🗣️ Echo — STT → TTS")
st.caption("Konuş → yazıya çevir → tekrar oku")

with st.expander("Ayarlar", expanded=False):
    orch_url = st.text_input(
        "Orchestrator /echo URL",
        value=DEFAULT_ORCH_URL,
        help="POST /echo kabul eden endpoint (örn: http://localhost:8010/echo)",
    )

st.write("**Konuş** düğmesine basıp mesajını kaydet. Bırakınca otomatik gönderilir; transcript ve ses yanıtı aşağıda görünür.")

# 1) Mikrofon kaydı
wav_bytes = audio_recorder(
    text="Konuş",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    sample_rate=44100,     # Backend 16kHz'e normalize edecek
    pause_threshold=1.0,   # 1 sn sessizlikte kaydı bitir
)

# 2) Alternatif: Dosyadan yükleme
with st.expander("Alternatif: WAV dosyası yükle", expanded=False):
    uploaded = st.file_uploader("WAV seç", type=["wav"])
    if uploaded and not wav_bytes:
        wav_bytes = uploaded.read()

# 3) Gönder ve sonucu göster
if wav_bytes:
    st.subheader("Orijinal kayıt")
    st.audio(wav_bytes, format="audio/wav")

    try:
        with st.spinner("Sunucuya gönderiliyor..."):
            files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
            resp = requests.post(orch_url or DEFAULT_ORCH_URL, files=files, timeout=120)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        st.error(f"İstek hatası: {e}")
        st.stop()

    text = data.get("text") or ""
    st.success("Transcript")
    st.markdown(f"> **{text}**")

    audio = (data or {}).get("audio") or {}
    b64 = audio.get("data")
    if isinstance(b64, str):
        try:
            out_wav = base64.b64decode(b64)
            st.subheader("Geri okunan ses")
            st.audio(out_wav, format="audio/wav")
        except Exception:
            st.warning("Geri dönen ses çözümlenemedi.")
    else:
        st.warning("Yanıtta ses verisi bulunamadı.")
