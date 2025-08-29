# VoiceRAGTR (voicerag-tr)

Türkçe odaklı **sesli çağrı asistanı** için **RAG (Retrieval-Augmented Generation)** temelli bilgi katmanı ve ajan mimarisi.  
Bu repo **paket paket** ilerleyecek; önce **RAG** katmanını kurup test ediyoruz, ardından **agent → tts → asr** ile zinciri tamamlayacağız.

> Hedef: Tamamen açık kaynak bileşenlerle (Qdrant + Sentence-Transformers vb.) çalışan, CPU/GPU’da koşabilen, modüler bir yapı.

---

## Yol Haritası (yüksek seviye)

- [x] Proje adı & repo
- [x] Paket 1: TTS (Piper/Coqui TTS; Türkçe ses)
- [x] Paket 2: ASR (faster-whisper; Türkçe konuşma tanıma)
- [ ] **Paket 3: Echo** (Orchestrator paketiyle streamlit kullanarak basit bir konuştuğunu dinle)
- [ ] Paket 4: RAG
  - [ ] Dizinler: `services/rag/`
  - [ ] FastAPI servisi: `/ingest`, `/search`, `/healthz`, `/readyz`, `/version`, `/metrics`
  - [ ] Qdrant ile vektör arama
  - [ ] `bge-m3` embedder ile çok dilli gömme
  - [ ] Örnek ingest & arama testleri
- [ ] Paket 5: Agent (LLM + tool-calling; RAG’i kullanarak **yalnızca kaynaklı** yanıt üretimi)
- [ ] (Opsiyonel) Gateway (WebRTC/SIP/WS köprüsü)
- [ ] CI/CD (GitHub Actions), gözlemlenebilirlik ve kalite

---

## Neden?

- **Modülerlik:** Her parça bağımsız bir servis (FastAPI) olarak çalışır.
- **Açık kaynak:** Maliyet kontrolü, özelleştirilebilirlik, denetlenebilirlik.
- **TR odaklı:** Türkçe ASR/TTS ve çok dilli RAG ile pratik sonuçlar.

---

## Önerilen Dizin Yapısı (adım adım oluşturacağız)

```text
voicerag-tr/
├─ services/
│  └─ rag/
│     ├─ app/            # FastAPI uygulaması
│     ├─ data/           # örnek belgeler
│     ├─ scripts/        # ingest gibi CLI yardımcıları
│     ├─ requirements.txt
│     └─ Dockerfile
├─ infra/
│  └─ docker-compose.yml # qdrant + rag servisi
├─ tests/
│  └─ rag/               # birim/entegrasyon testleri
└─ README.md
