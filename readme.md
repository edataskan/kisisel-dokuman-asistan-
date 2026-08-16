 Mimari

- **Embedding**: `sentence-transformers` ile local çalışır (`all-MiniLM-L6-v2`) — harici API'ye bağımlı değil
- **Retrieval**: numpy ile cosine similarity (vektör veritabanı yok, brute-force arama)
- **Generation**: Google Gemini API (`gemini-flash-latest`)

## Kurulum

```bash
pip install -r requirements.txt
docker compose up -d 
```
API key al
PowerShell'de key'i tanımla:

```powershell
$env:GEMINI_API_KEY = ""
```

## Dosyalar

rag.py **Adım 1** — Her dosya tek parça olarak embed edilir, retrieval dosya seviyesinde |
rag_v2_chunking.py **Adım 2** — Dosyalar 500 karakterlik, 50 karakter overlap'li parçalara (chunk) bölünür, retrieval chunk seviyesinde |
rag_v3_pgvector.py	Adım 3 — Chunk'lar numpy yerine Postgres + pgvector'da saklanır, similarity SQL ile hesaplanır
rag_v4_hybrid.py	Adım 4 — Vector search + BM25 (RRF ile birleştirilir) + cross-encoder re-ranking
| `notlar/` | Örnek `.txt` dokümanları — kendi notlarınla değiştirebilirsin |

## Çalıştırma

```bash
python rag.py                  # adım 1
python rag_v2_chunking.py      # adım 2 (chunking)
python rag_v3_pgvector.py --index    # dokümanları oku, embed et, DB'ye yaz (bir kere yeter)
python rag_v3_pgvector.py            # adım 3: pgvector ile soru-cevap
python rag_v4_hybrid.py              # adım 4: hybrid search + re-ranking ile soru-cevap
```

Sonra bir soru sor, sistem en alakalı kaynağı/kaynakları bulup Gemini'ye context olarak verir
ve cevabı üretir.

## Yol Haritası

- [x] Adım 1 — Temel RAG (embedding + similarity + generation)
- [x] Adım 2 — Chunking (overlap'li sabit boyutlu parçalama)
- [x] Adım 3 — Vektör veritabanı (Qdrant / pgvector) — brute-force numpy aramasının yerini alacak
- [x] Adım 4 — Hybrid search (BM25 + vector) ve re-ranking
- [ ] Adım 5 — Evaluation (RAGAS ile faithfulness/relevance ölçümü)
- [ ] Adım 6 — Agent katmanı (tool use, ReAct pattern, memory)
- [ ] Adım 7 — FastAPI servisi + Docker + Kafka/Redis ile prodüksiyona yaklaştırma


