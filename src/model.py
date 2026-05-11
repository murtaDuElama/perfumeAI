"""
PerfumeAI – Model Modülü (Kişi 2)
Görevler:
  - Kullanıcı sorgusunu TF-IDF vektörüne çevirme
  - Kosinüs benzerliği ile parfüm önerisi
  - Marka, koku ailesi ve parfüm tipi bazlı filtreleme
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# Yardımcı fonksiyon
# ──────────────────────────────────────────────

def _sorguyu_vektorlestir(sorgu: str, vectorizer):
    """
    Kullanıcıdan gelen serbest metin sorgusunu TF-IDF vektörüne çevirir.

    Parametreler
    ------------
    sorgu : str
        Örnek: "bergamot ve gül içeren, odunsu bir parfüm"
    vectorizer : TfidfVectorizer
        veri.py'nin build_tfidf() fonksiyonundan dönen eğitilmiş vektörleştirici.

    Döndürür
    --------
    scipy sparse matrix – (1 × n_özellik) boyutunda sorgu vektörü.
    """
    # küçük harfe çeviriyoruz çünkü veri.py'de tüm notalar küçük harfle kaydedildi
    # sorguda büyük harf varsa eşleşme bulunamaz
    sorgu_temiz = sorgu.strip().lower()

    # transform kullanıyoruz (fit_transform değil!) çünkü vectorizer zaten eğitilmiş durumda
    # fit_transform burada yanlış olur, veri seti kelimelerini sıfırlar
    return vectorizer.transform([sorgu_temiz])


def _filtrele(df: pd.DataFrame,
              koku_ailesi: str | None = None,
              marka: str | None = None,
              tip: str | None = None) -> np.ndarray:
    """
    Verilen kriterlere göre DataFrame'i filtreler ve uygun satırların
    konumsal (positional) index dizisini döndürür.

    Parametreler
    ------------
    df          : Tam parfüm DataFrame'i (load_data() çıktısı).
    koku_ailesi : Örn. "Floral", "Woody" – büyük/küçük harf duyarsız.
    marka       : Örn. "Chanel" – büyük/küçük harf duyarsız.
    tip         : Örn. "Eau de Parfum" – büyük/küçük harf duyarsız.

    Döndürür
    --------
    np.ndarray – filtrelenmiş satırların konumsal (0-tabanlı) index dizisi.
    """
    # başlangıçta tüm satırları seçili kabul et, filtreler uygulandıkça daralt
    maske = np.ones(len(df), dtype=bool)

    # her filtre parametresi opsiyonel, None gelirse o filtreyi atlıyoruz
    # karşılaştırmayı lower() ile yapıyoruz ki büyük/küçük harf fark etmesin
    if koku_ailesi:
        maske &= df["koku_ailesi"].str.lower().values == koku_ailesi.lower()

    if marka:
        maske &= df["marka"].str.lower().values == marka.lower()

    if tip:
        maske &= df["tip"].str.lower().values == tip.lower()

    # True olan konumların index numaralarını döndür (TF-IDF matrisinde dilim almak için gerekiyor)
    return np.where(maske)[0]


# ──────────────────────────────────────────────
# Ana öneri fonksiyonu
# ──────────────────────────────────────────────

def recommend(
    sorgu: str,
    df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    top_n: int = 5,
    koku_ailesi: str | None = None,
    marka: str | None = None,
    tip: str | None = None,
) -> pd.DataFrame:
    """
    Kullanıcı sorgusuna en benzer parfümleri kosinüs benzerliğiyle bulur.

    Parametreler
    ------------
    sorgu        : str  – Kullanıcının doğal dil sorgusu.
                          Örn: "taze narenciye ve lavanta"
    df           : pd.DataFrame – veri.py'nin load_data() çıktısı.
    vectorizer   : TfidfVectorizer – build_tfidf() çıktısı.
    tfidf_matrix : sparse matrix   – build_tfidf() çıktısı.
    top_n        : int  – Kaç öneri döndürülsün? (varsayılan 5)
    koku_ailesi  : str  – Filtre: koku ailesi (ör. "Floral"). Opsiyonel.
    marka        : str  – Filtre: marka adı (ör. "Dior"). Opsiyonel.
    tip          : str  – Filtre: parfüm tipi (ör. "Eau de Parfum"). Opsiyonel.

    Döndürür
    --------
    pd.DataFrame – Sütunlar: marka, ad, koku_ailesi, tip, benzerlik_skoru,
                              ust_nota, orta_nota, alt_nota
    """

    # 1 ── Kullanıcının yazdığı metni TF-IDF vektörüne çevir
    sorgu_vektoru = _sorguyu_vektorlestir(sorgu, vectorizer)

    # 2 ── Filtreleri uygula, hangi satırların dahil edileceğini bul
    filtre_index = _filtrele(df, koku_ailesi=koku_ailesi, marka=marka, tip=tip)

    # Eğer filtreye hiç parfüm uymadıysa uyarı ver ve filtresiz devam et
    # yoksa boş sonuç dönmek yerine en azından genel öneriler gösterebiliyoruz
    if len(filtre_index) == 0:
        print("[model] UYARI: Filtreye uyan parfüm bulunamadı. Filtresiz arama yapılıyor.")
        filtre_index = df.index

    # 3 ── Tüm matris yerine sadece filtreye uyan satırları al
    # bu hem daha hızlı hem de yanlış satırların önerilmesini engelliyor
    tfidf_alt_kume = tfidf_matrix[filtre_index]

    # 4 ── Kullanıcı sorgusu ile her parfüm arasındaki açısal benzerliği hesapla
    # 0 = hiç benzer değil, 1 = tamamen aynı
    benzerlikler = cosine_similarity(sorgu_vektoru, tfidf_alt_kume).flatten()

    # 5 ── Skorları büyükten küçüğe sırala ve ilk top_n tanesini al
    # argsort küçükten büyüğe sıralar, [::-1] ile ters çeviriyoruz
    en_iyi_sirali = np.argsort(benzerlikler)[::-1][:top_n]

    # 6 ── Bulunan konumsal indexleri gerçek DataFrame satırlarına çevir
    secilen_pozisyon = filtre_index[en_iyi_sirali]
    df_reset = df.reset_index(drop=True)   # güvenli iloc erişimi için index'i sıfırla
    sonuclar = df_reset.iloc[secilen_pozisyon, :].copy()

    # sadece kullanıcıya gösterilecek sütunları tut
    sonuclar = sonuclar[["marka", "ad", "koku_ailesi", "tip",
                          "ust_nota", "orta_nota", "alt_nota"]].copy()

    # hesaplanan benzerlik skorunu tabloya ekle, 4 basamak yeterli
    sonuclar["benzerlik_skoru"] = benzerlikler[en_iyi_sirali].round(4)
    sonuclar = sonuclar.reset_index(drop=True)

    # skor sütununu öne al ki kullanıcı kolayca görebilsin
    return sonuclar[["marka", "ad", "koku_ailesi", "tip",
                      "benzerlik_skoru", "ust_nota", "orta_nota", "alt_nota"]]


# ──────────────────────────────────────────────
# Hızlı test (python model.py ile çalıştır)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # model.py'yi doğrudan çalıştırınca veri.py'yi bulamıyor, path'e ekliyoruz
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from veri import load_data, build_tfidf

    df = load_data("data/parfumes.csv")
    vectorizer, tfidf_matrix = build_tfidf(df)

    # ── Test 1: Serbest sorgu ──
    print("\n" + "="*55)
    print("TEST 1 – Serbest sorgu: 'bergamot ve misk'")
    print("="*55)
    sonuc = recommend("bergamot ve misk", df, vectorizer, tfidf_matrix, top_n=5)
    print(sonuc.to_string(index=False))

    # ── Test 2: Koku ailesi filtresi ──
    print("\n" + "="*55)
    print("TEST 2 – Filtreyle: 'çiçeksi ve taze', koku_ailesi=Floral")
    print("="*55)
    sonuc2 = recommend(
        "çiçeksi ve taze",
        df, vectorizer, tfidf_matrix,
        top_n=5,
        koku_ailesi="Floral"
    )
    print(sonuc2.to_string(index=False))

    # ── Test 3: Marka filtresi ──
    print("\n" + "="*55)
    print("TEST 3 – Filtreyle: 'odunsu', marka=Chanel")
    print("="*55)
    sonuc3 = recommend(
        "odunsu",
        df, vectorizer, tfidf_matrix,
        top_n=5,
        marka="Chanel"
    )
    print(sonuc3.to_string(index=False))