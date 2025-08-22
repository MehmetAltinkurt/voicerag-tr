# services/rag/app/main.py
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from prometheus_fastapi_instrumentator import Instrumentator

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# Token-doğru parçalama (bge-m3 tokenizer + LangChain splitter)
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================
# Ayarlar (env'den okunur)
# =============================
class Settings(BaseSettings):
    SERVICE_NAME: str = "rag"
    VERSION: str = "0.2.0"

    # Qdrant bağlantısı
    QDRANT_URL: str = "http://qdrant:6333"
    COLLECTION_NAME: str = "docs"

    # Embed modeli ve parçalama ayarları (token cinsinden)
    EMBED_MODEL: str = "BAAI/bge-m3"
    MAX_CHUNK_TOKENS: int = 500
    CHUNK_OVERLAP: int = 100

    class Config:
        env_file = ".env"


settings = Settings()
app = FastAPI(title="VoiceRAGTR - RAG Service", version=settings.VERSION)

READY = False

# Global bileşenler
_embedder: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None
_vector_size: Optional[int] = None
_tokenizer: Optional[AutoTokenizer] = None
_splitter: Optional[RecursiveCharacterTextSplitter] = None


# =============================
# Veri şemaları
# =============================
class IngestDoc(BaseModel):
    id: Optional[str] = None
    text: str
    source: Optional[str] = None
    metadata: Optional[dict] = None


class IngestRequest(BaseModel):
    documents: List[IngestDoc]
    chunk_tokens: Optional[int] = Field(default=None, description="Varsayılanı override etmek için")
    chunk_overlap: Optional[int] = Field(default=None, description="Varsayılanı override etmek için")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchChunk(BaseModel):
    id: str
    text: str
    source: Optional[str] = None
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    chunks: List[SearchChunk]


# =============================
# Yardımcılar
# =============================
def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """
    Koleksiyon yoksa oluşturur, varsa dokunmaz (idempotent).
    """
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _hf_len(text: str) -> int:
    """
    HF tokenizer ile token sayımı (bge-m3 ile aynı tokenizer).
    """
    assert _tokenizer is not None, "Tokenizer not initialized"
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _build_splitter(max_tokens: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Paragraf -> cümle -> kelime -> karakter sıralı ayırma;
    length_function olarak HF tokenizer kullanır (token-doğru).
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=_hf_len,
    )


def _chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    Token-doğru parçalama: Varsayılan ayarlar kullanılıyorsa global splitter,
    override gelmişse geçici splitter oluşturur.
    """
    global _splitter
    if (
        max_tokens == settings.MAX_CHUNK_TOKENS
        and overlap == settings.CHUNK_OVERLAP
        and _splitter is not None
    ):
        return _splitter.split_text(text)

    # Override edilirse anlık splitter kur
    tmp_splitter = _build_splitter(max_tokens, overlap)
    return tmp_splitter.split_text(text)


# =============================
# Yaşam döngüsü
# =============================
@app.on_event("startup")
def on_startup():
    """
    - Prometheus metriklerini aç
    - Embedding modelini yükle
    - bge-m3 tokenizer'ını yükle
    - Qdrant'a bağlan, koleksiyonu hazırla
    """
    global READY, _embedder, _qdrant, _vector_size, _tokenizer, _splitter

    # /metrics endpoint'ini otomatik ekler
    Instrumentator().instrument(app).expose(app)

    # Embedder
    _embedder = SentenceTransformer(settings.EMBED_MODEL)
    _vector_size = _embedder.get_sentence_embedding_dimension()

    # Tokenizer (token-doğru chunking için)
    _tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)

    # Varsayılan splitter (global)
    _splitter = _build_splitter(settings.MAX_CHUNK_TOKENS, settings.CHUNK_OVERLAP)

    # Qdrant bağlantısı
    _qdrant = QdrantClient(url=settings.QDRANT_URL)
    _ensure_collection(_qdrant, settings.COLLECTION_NAME, _vector_size)

    READY = True


# =============================
# Sağlık & bilgi
# =============================
@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.SERVICE_NAME}


@app.get("/readyz")
def readyz():
    return {"ready": READY}


@app.get("/version")
def version():
    return {"service": settings.SERVICE_NAME, "version": settings.VERSION}


# =============================
# API: Ingest
# =============================
@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    Dokümanları chunk'lara bölüp embed'leyerek Qdrant'a yazar.
    """
    if not READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    max_tokens = req.chunk_tokens or settings.MAX_CHUNK_TOKENS
    overlap = req.chunk_overlap or settings.CHUNK_OVERLAP

    ids: List[str] = []
    texts: List[str] = []
    payloads: List[dict] = []

    for d in req.documents:
        doc_id = d.id or str(uuid.uuid4())
        for idx, ch in enumerate(_chunk_text(d.text, max_tokens, overlap)):
            cid = f"{doc_id}::{idx}"
            ids.append(cid)
            texts.append(ch)
            payloads.append(
                {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "source": d.source,
                    "metadata": d.metadata or {},
                    "text": ch,
                }
            )

    if not texts:
        return {"ingested": 0}

    # Normalize edilmiş embedding (cosine için iyi pratik)
    embs = _embedder.encode(texts, normalize_embeddings=True)
    vectors = [list(map(float, e)) for e in embs]

    _qdrant.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
    )
    return {"ingested": len(ids)}


# =============================
# API: Search
# =============================
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Sorguyu embed'leyip vektör araması yapar, en iyi eşleşmeleri döner.
    """
    if not READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    q_emb = _embedder.encode([req.query], normalize_embeddings=True)[0].tolist()

    res = _qdrant.search(
        collection_name=settings.COLLECTION_NAME,
        query_vector=q_emb,
        limit=max(1, min(50, req.top_k)),
        with_payload=True,
    )

    chunks: List[SearchChunk] = []
    for p in res:
        payload = p.payload or {}
        chunks.append(
            SearchChunk(
                id=str(p.id),
                text=payload.get("text", ""),
                source=payload.get("source"),
                score=float(p.score) if p.score is not None else 0.0,  # cosine similarity
                metadata=payload.get("metadata"),
            )
        )
    return SearchResponse(chunks=chunks)
