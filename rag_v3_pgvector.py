"""
Sifirdan RAG - Adim 3: Vektor Veritabani (pgvector)
Adim 2'ye ek olarak: chunklar artik numpy array'de degil, Postgres + pgvector'da saklanir.
Cosine similarity hesabi artik Python'da degil, SQL sorgusuyla veritabaninda yapilir.

Kurulum:
    pip install -r requirements.txt
    docker compose up -d          # postgres + pgvector'i ayaga kaldirir

Calistirma:
    python rag_v3_pgvector.py --index     # notlar/ klasorunu oku, embed et, DB'ye yaz
    python rag_v3_pgvector.py             # soru-cevap moduna gir (indexleme yapilmis olmali)
"""

import os
import sys
import glob
import numpy as np
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-flash-latest"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_BOYUTU = 384  

NOTLAR_KLASORU = "notlar"
CHUNK_BOYUTU = 500
CHUNK_OVERLAP = 50

DB_URL = "postgresql://rag:rag@127.0.0.1:5433/rag"


def metni_parcala(metin: str, boyut: int = CHUNK_BOYUTU, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(metin) <= boyut:
        return [metin]
    parcalar = []
    baslangic = 0
    while baslangic < len(metin):
        bitis = baslangic + boyut
        parca = metin[baslangic:bitis].strip()
        if parca:
            parcalar.append(parca)
        baslangic += boyut - overlap
    return parcalar

def db_baglan():
    return psycopg.connect(DB_URL, row_factory=dict_row)


def tabloyu_hazirla():
    with db_baglan() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunklar (
                id SERIAL PRIMARY KEY,
                dosya TEXT NOT NULL,
                chunk_no INT NOT NULL,
                metin TEXT NOT NULL,
                embedding vector({EMBEDDING_BOYUTU}) NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunklar_embedding_idx
            ON chunklar USING hnsw (embedding vector_cosine_ops)
        """)
        conn.commit()


def indexle():
    tabloyu_hazirla()
    
    with db_baglan() as conn:
        conn.execute("TRUNCATE TABLE chunklar")
        conn.commit()
        
        toplam = 0
        for path in glob.glob(os.path.join(NOTLAR_KLASORU, "*.txt")):
            dosya_adi = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                metin = f.read()
            parcalar = metni_parcala(metin)
            vektorler = embed_model.encode(parcalar, normalize_embeddings=True)
            
            for i, (parca, vektor) in enumerate(zip(parcalar, vektorler)):
                conn.execute(
                    "INSERT INTO chunklar (dosya, chunk_no, metin, embedding) VALUES (%s, %s, %s, %s)",
                    (dosya_adi, i, parca, vektor.tolist()),
                )
            conn.commit()
            toplam += len(parcalar)
            print(f"  {dosya_adi}: {len(parcalar)} chunk indexlendi")
            
        print(f"\nToplam {toplam} chunk veritabanina yazildi.")

def en_yakin_chunklari_bul(soru: str, k: int = 3) -> list[dict]:
    soru_vektoru = embed_model.encode([soru], normalize_embeddings=True)[0]

    with db_baglan() as conn:
        sonuclar = conn.execute(
            """
            SELECT dosya, chunk_no, metin, 1 - (embedding <=> %s) AS benzerlik
            FROM chunklar
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (soru_vektoru.tolist(), soru_vektoru.tolist(), k),
        ).fetchall()

    return sonuclar

def cevapla(soru: str, context: str) -> str:
    sistem_talimati = (
        "Sana verilen CONTEXT icindeki bilgiye dayanarak soruyu cevapla. "
        "Eger context'te cevap yoksa, 'Bu bilgi verilen dokumanlarda yok' de. "
        "Uydurma bilgi verme."
    )
    yanit = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"CONTEXT:\n{context}\n\nSORU: {soru}",
        config={"system_instruction": sistem_talimati, "max_output_tokens": 500},
    )
    return yanit.text


def soru_cevap_modu():
    while True:
        soru = input("\nSoru (cikmak icin 'q'): ").strip()
        if soru.lower() == "q":
            break

        sonuclar = en_yakin_chunklari_bul(soru, k=3)
        if not sonuclar:
            print("Veritabaninda hic chunk yok. Once '--index' ile indexleme yap.")
            continue

        print("\n--- Bulunan kaynaklar ---")
        secilen_metinler = []
        for s in sonuclar:
            print(f"  {s['dosya']} [chunk {s['chunk_no']}]  (benzerlik: {s['benzerlik']:.3f})")
            secilen_metinler.append(s["metin"])

        context = "\n\n---\n\n".join(secilen_metinler)
        cevap = cevapla(soru, context)
        print(f"\n--- Cevap ---\n{cevap}")


def main():
    if "--index" in sys.argv:
        print("Dokumanlar indexleniyor...")
        indexle()
    else:
        tabloyu_hazirla()
        soru_cevap_modu()


if __name__ == "__main__":
    main()
