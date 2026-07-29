Framework kullanmadan (LangChain, LlamaIndex vb. yok), RAG'in temel mekanizmasını anlamak için
adım adım geliştirilen bir öğrenme projesi. Embedding + cosine similarity + LLM zincirinin
altında ne olduğunu görmek amacıyla her şey ham Python ile yazıldı.

## Mimari

- **Embedding**: `sentence-transformers` ile local çalışır (`all-MiniLM-L6-v2`) — harici API'ye bağımlı değil
- **Retrieval**: numpy ile cosine similarity (vektör veritabanı yok, brute-force arama)
- **Generation**: Google Gemini API (`gemini-flash-latest`)

## Kurulum

```bash
pip install -r requirements.txt
```

API key al: [aistudio.google.com](https://aistudio.google.com) → "Get API Key" (kredi kartı istemez)

PowerShell'de key'i tanımla:

```powershell
$env:GEMINI_API_KEY = "senin-key-in"
```

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `rag.py` | **Adım 1** — Her dosya tek parça olarak embed edilir, retrieval dosya seviyesinde |
| `rag_v2_chunking.py` | **Adım 2** — Dosyalar 500 karakterlik, 50 karakter overlap'li parçalara (chunk) bölünür, retrieval chunk seviyesinde |
| `notlar/` | Örnek `.txt` dokümanları — kendi notlarınla değiştirebilirsin |

## Çalıştırma

```bash
python rag.py                  # adım 1
python rag_v2_chunking.py      # adım 2 (chunking)
```

Sonra bir soru sor, sistem en alakalı kaynağı/kaynakları bulup Gemini'ye context olarak verir
ve cevabı üretir.

## Yol Haritası

- [x] Adım 1 — Temel RAG (embedding + similarity + generation)
- [x] Adım 2 — Chunking (overlap'li sabit boyutlu parçalama)
- [ ] Adım 3 — Vektör veritabanı (Qdrant / pgvector) — brute-force numpy aramasının yerini alacak
- [ ] Adım 4 — Hybrid search (BM25 + vector) ve re-ranking
- [ ] Adım 5 — Evaluation (RAGAS ile faithfulness/relevance ölçümü)
- [ ] Adım 6 — Agent katmanı (tool use, ReAct pattern, memory)
- [ ] Adım 7 — FastAPI servisi + Docker + Kafka/Redis ile prodüksiyona yaklaştırma

## Notlar

- `CHUNK_BOYUTU` ve `CHUNK_OVERLAP` değerleri `rag_v2_chunking.py` içinde ayarlanabilir
- `notlar/` klasörüne `.txt` dışında dosya koyma — script tüm `.txt` dosyalarını doküman olarak okur
