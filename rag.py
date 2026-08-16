import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-flash-latest"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  

NOTLAR_KLASORU = "notlar"


def dokumanlari_yukle(klasor: str) -> list[dict]:
    dokumanlar = []
    for path in glob.glob(os.path.join(klasor, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            metin = f.read()
        dokumanlar.append({"dosya": os.path.basename(path), "metin": metin})
    return dokumanlar


def embed_et(metinler: list[str]) -> np.ndarray:
    return embed_model.encode(metinler, normalize_embeddings=True)


def en_yakin_dokumani_bul(soru_vektoru: np.ndarray, doc_vektorleri: np.ndarray, k: int = 1):
    skorlar = doc_vektorleri @ soru_vektoru
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
    print("Dokumanlar yukleniyor...")
    dokumanlar = dokumanlari_yukle(NOTLAR_KLASORU)
    if not dokumanlar:
        print(f"'{NOTLAR_KLASORU}' klasorunde .txt dosyasi bulunamadi.")
        return

    metinler = [d["metin"] for d in dokumanlar]
    print(f"{len(metinler)} dokuman bulundu, embedding hesaplaniyor...")
    doc_vektorleri = embed_et(metinler)

    while True:
        soru = input("\nSoru (cikmak icin 'q'): ").strip()
        if soru.lower() == "q":
            break

        soru_vektoru = embed_et([soru])[0]
        indeksler, skorlar = en_yakin_dokumani_bul(soru_vektoru, doc_vektorleri, k=2)

        print("\n--- Bulunan kaynaklar ---")
        secilen_metinler = []
        for idx, skor in zip(indeksler, skorlar):
            print(f"  {dokumanlar[idx]['dosya']}  (benzerlik: {skor:.3f})")
            secilen_metinler.append(dokumanlar[idx]["metin"])

        context = "\n\n---\n\n".join(secilen_metinler)
        cevap = cevapla(soru, context)
        print(f"\n--- Cevap ---\n{cevap}")


if __name__ == "__main__":
    main()