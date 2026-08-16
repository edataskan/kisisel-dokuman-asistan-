import os
import sys
import glob
import re
import numpy as np
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from google import genai


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-flash-latest"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_BOYUTU = 384

NOTLAR_KLASORU = "notlar"
CHUNK_BOYUTU = 500
CHUNK_OVERLAP = 50

DB_URL = "postgresql://rag:rag@localhost:5432/rag"

VECTOR_K = 10   # vector search'ten kac aday alinacak
BM25_K = 10     # BM25'ten kac aday alinacak
RERANK_K = 3    # re-ranking sonrasi kac sonuc kullanilacak
RRF_K = 60      # RRF formulundeki sabit (standart deger)

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


def tokenize(metin: str) -> list[str]:
    return re.findall(r"\w+", metin.lower())


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

        with db_baglan() as conn:
            for i, (parca, vektor) in enumerate(zip(parcalar, vektorler)):
                conn.execute(
                    "INSERT INTO chunklar (dosya, chunk_no, metin, embedding) VALUES (%s, %s, %s, %s)",
                    (dosya_adi, i, parca, vektor.tolist()),
                )
            conn.commit()

        toplam += len(parcalar)
        print(f"  {dosya_adi}: {len(parcalar)} chunk indexlendi")

    print(f"\nToplam {toplam} chunk veritabanina yazildi.")


def tum_chunklari_getir() -> list[dict]:
    with db_baglan() as conn:
        return conn.execute("SELECT id, dosya, chunk_no, metin FROM chunklar").fetchall()


def vector_search(soru: str, k: int = VECTOR_K) -> list[dict]:
    soru_vektoru = embed_model.encode([soru], normalize_embeddings=True)[0]
    with db_baglan() as conn:
        return conn.execute(
            """
            SELECT id, dosya, chunk_no, metin, 1 - (embedding <=> %s) AS benzerlik
            FROM chunklar
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (soru_vektoru.tolist(), soru_vektoru.tolist(), k),
        ).fetchall()


def bm25_search(soru: str, tum_chunklar: list[dict], k: int = BM25_K) -> list[dict]:
    corpus = [tokenize(c["metin"]) for c in tum_chunklar]
    bm25 = BM25Okapi(corpus)
    skorlar = bm25.get_scores(tokenize(soru))

    en_iyi_indeksler = np.argsort(skorlar)[::-1][:k]
    sonuclar = []
    for idx in en_iyi_indeksler:
        if skorlar[idx] <= 0:
            continue  # hic eslesme yoksa alma
        c = tum_chunklar[idx]
        sonuclar.append({**c, "bm25_skoru": float(skorlar[idx])})
    return sonuclar


def rrf_birlestir(vector_sonuclari: list[dict], bm25_sonuclari: list[dict], k_sabit: int = RRF_K) -> list[dict]:
    
    rrf_skorlari = {}
    chunk_verisi = {}

    for sira, c in enumerate(vector_sonuclari):
        rrf_skorlari[c["id"]] = rrf_skorlari.get(c["id"], 0) + 1 / (k_sabit + sira)
        chunk_verisi[c["id"]] = c

    for sira, c in enumerate(bm25_sonuclari):
        rrf_skorlari[c["id"]] = rrf_skorlari.get(c["id"], 0) + 1 / (k_sabit + sira)
        chunk_verisi[c["id"]] = c

    siralanmis_idler = sorted(rrf_skorlari, key=rrf_skorlari.get, reverse=True)
    return [chunk_verisi[cid] for cid in siralanmis_idler]


def rerank(soru: str, adaylar: list[dict], k: int = RERANK_K) -> list[dict]:
    if not adaylar:
        return []

    ciftler = [(soru, c["metin"]) for c in adaylar]
    skorlar = reranker.predict(ciftler)  # yuksek skor = daha alakali

    for c, skor in zip(adaylar, skorlar):
        c["rerank_skoru"] = float(skor)

    siralanmis = sorted(adaylar, key=lambda c: c["rerank_skoru"], reverse=True)
    return siralanmis[:k]


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
    tum_chunklar = tum_chunklari_getir()
    if not tum_chunklar:
        print("Veritabaninda hic chunk yok. Once '--index' ile indexleme yap.")
        return

    while True:
        soru = input("\nSoru (cikmak icin 'q'): ").strip()
        if soru.lower() == "q":
            break

        vec_sonuc = vector_search(soru, k=VECTOR_K)
        bm25_sonuc = bm25_search(soru, tum_chunklar, k=BM25_K)
        birlesik = rrf_birlestir(vec_sonuc, bm25_sonuc)
        final = rerank(soru, birlesik, k=RERANK_K)

        print("\n--- Bulunan kaynaklar (re-rank sonrasi) ---")
        secilen_metinler = []
        for c in final:
            print(f"  {c['dosya']} [chunk {c['chunk_no']}]  (rerank skoru: {c['rerank_skoru']:.3f})")
            secilen_metinler.append(c["metin"])

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
