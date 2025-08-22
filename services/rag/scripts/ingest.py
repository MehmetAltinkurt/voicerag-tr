# services/rag/scripts/ingest.py
import argparse
import glob
import os
import sys
import requests


def read_text(path: str) -> str:
    """UTF-8 ağırlıklı güvenli okuma (bozuk karakterleri yoksayar)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RAG servisine dosyaları ingest eder (FastAPI /ingest)."
    )
    ap.add_argument(
        "--base-url",
        required=True,
        help="RAG servis kök adresi, örn: http://localhost:8003",
    )
    ap.add_argument(
        "--glob",
        default="services/rag/data/*.md",
        help="İçeri alınacak dosyalar için glob deseni (örn: 'services/rag/data/*.md')",
    )
    ap.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="Varsayılan MAX_CHUNK_TOKENS değerini bu çağrı için override et",
    )
    ap.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Varsayılan CHUNK_OVERLAP değerini bu çağrı için override et",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="İsteğe bağlı; en fazla şu kadar dosya al (güvenli test için)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="İstek atma, sadece hangi dosyaların yükleneceğini göster",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if args.max_files is not None:
        files = files[: max(0, args.max_files)]

    if not files:
        print("Uyarı: Glob desenine uyan dosya bulunamadı:", args.glob)
        return 1

    docs = []
    for fp in files:
        text = read_text(fp)
        doc_id = os.path.splitext(os.path.basename(fp))[0]
        docs.append(
            {
                "id": doc_id,
                "text": text,
                "source": fp,
                "metadata": {"ext": os.path.splitext(fp)[1]},
            }
        )

    print(f"Toplam dosya: {len(docs)}")
    for d in docs[:5]:
        print(f"  - {d['id']} (kaynak: {d['source']})")
    if len(docs) > 5:
        print(f"  ... (+{len(docs) - 5} dosya)")

    if args.dry_run:
        print("Dry-run bitti: istek gönderilmedi.")
        return 0

    payload = {"documents": docs}
    if args.chunk_tokens is not None:
        payload["chunk_tokens"] = args.chunk_tokens
    if args.chunk_overlap is not None:
        payload["chunk_overlap"] = args.chunk_overlap

    base = args.base_url.rstrip("/")
    url = f"{base}/ingest"

    try:
        # kısa bağlantı + uzun işlem zaman aşımı
        resp = requests.post(url, json=payload, timeout=(5, 300))
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Hata: /ingest isteği başarısız:", e, file=sys.stderr)
        return 2

    print("Sunucu yanıtı:", resp.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
