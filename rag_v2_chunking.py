import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# ---- 1. Setup ----

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-flash-latest"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

NOTLAR_KLASORU = "notlar"
CHUNK_BOYUTU = 500      # karakter
CHUNK_OVERLAP = 50      # karakter


# ---- 2. Chunking ----

def metni_parcala(metin: str, boyut: int = CHUNK_BOYUTU, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Metni sabit boyutlu, ortusen parcalara boler.
    Ornek: boyut=500, overlap=50 -> her parca 500 karakter,
    bir sonraki parca 450. karakterden baslar (son 50 karakter ortusur).
    """
    if len(metin) <= boyut:
        return [metin]

    parcalar = []
    baslangic = 0
    while baslangic < len(metin):
        bitis = baslangic + boyut
        parca = metin[baslangic:bitis].strip()
        if parca:
            parcalar.append(parca)
        baslangic += boyut - overlap  # overlap kadar geri git

    return parcalar


def dokumanlari_yukle_ve_parcala(klasor: str) -> list[dict]:
    chunklar = []
    for path in glob.glob(os.path.join(klasor, "*.txt")):
        dosya_adi = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            metin = f.read()

        parcalar = metni_parcala(metin)
        for i, parca in enumerate(parcalar):
            chunklar.append({
                "dosya": dosya_adi,
                "chunk_no": i,
                "metin": parca,
            })

    return chunklar



def embed_et(metinler: list[str]) -> np.ndarray:
    return embed_model.encode(metinler, normalize_embeddings=True)


def en_yakin_chunklari_bul(soru_vektoru: np.ndarray, chunk_vektorleri: np.ndarray, k: int = 3):
    skorlar = chunk_vektorleri @ soru_vektoru
    en_iyi_indeksler = np.argsort(skorlar)[::-1][:k]
    return en_iyi_indeksler, skorlar[en_iyi_indeksler]


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


def main():
    print("Dokumanlar yukleniyor ve parcalaniyor...")
    chunklar = dokumanlari_yukle_ve_parcala(NOTLAR_KLASORU)
    if not chunklar:
        print(f"'{NOTLAR_KLASORU}' klasorunde .txt dosyasi bulunamadi.")
        return

    metinler = [c["metin"] for c in chunklar]
    print(f"{len(chunklar)} chunk olusturuldu, embedding hesaplaniyor...")
    chunk_vektorleri = embed_et(metinler)

    while True:
        soru = input("\nSoru (cikmak icin 'q'): ").strip()
        if soru.lower() == "q":
            break

        soru_vektoru = embed_et([soru])[0]
        indeksler, skorlar = en_yakin_chunklari_bul(soru_vektoru, chunk_vektorleri, k=3)

        print("\n--- Bulunan kaynaklar ---")
        secilen_metinler = []
        for idx, skor in zip(indeksler, skorlar):
            c = chunklar[idx]
            print(f"  {c['dosya']} [chunk {c['chunk_no']}]  (benzerlik: {skor:.3f})")
            secilen_metinler.append(c["metin"])

        context = "\n\n---\n\n".join(secilen_metinler)
        cevap = cevapla(soru, context)
        print(f"\n--- Cevap ---\n{cevap}")


if __name__ == "__main__":
    main()
