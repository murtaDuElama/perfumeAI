"""
veri.py - PerfumeAI Veri Modulu
================================
Kaynak  : parfumes_hf.csv (pelegelraz/perfumes-dataset)
Satirlar: 10.000 parfum, her koku ailesinden 300 ornek alinir (toplam 3000)

Fonksiyonlar:
    load_data(csv_yolu)  -> pd.DataFrame
    build_tfidf(df)      -> (TfidfVectorizer, sparse matrix)
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------
# Sabitler
# --------------------------------------------------------------

# CSV dosyasinin varsayilan konumu - bu dosyayi calistirdigimiz yerden
# bagimsiz olarak her zaman dogru yolu bulmak icin __file__ kullandim
VARSAYILAN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "parfumes.csv")

# Her koku ailesinden kac kayit alacagimizi burada belirliyoruz
# 10 aile x 300 = 3000 satirlik dengeli bir veri seti olusturuyor
AILE_BASI_KAYIT = 300

# orijinal CSV Ingilizce sutun adlari kullaniyor, bunlari Turkce'ye cevirdim
# boylece model.py ve app.py'de daha anlasilir kodlar yazabiliyoruz
SUTUN_ESLEME = {
    "perfume_id"   : "parfum_id",
    "brand"        : "marka",
    "perfume_name" : "ad",
    "full_name"    : "tam_ad",
    "perfume_type" : "tip",
    "family"       : "koku_ailesi",
    "top_notes"    : "ust_nota",
    "middle_notes" : "orta_nota",
    "base_notes"   : "alt_nota",
    "occasions"    : "kullanim",
    "moods"        : "ruh_hali",
    "all_notes"    : "tum_notalar_ham",
}


# --------------------------------------------------------------
# Yardimci fonksiyon
# --------------------------------------------------------------
def _normalize_notalar(metin: str) -> str:
    """
    Virgülle ayrılmış nota stringini temizler:
    - Kucuk harfe cevirir
    - Bos ve tekrar eden notalari kaldirir
    - Boslukla ayrilmis tek string dondurur

    Ornek:
        "Gul , Yasemin,  gul" -> "gul yasemin"
    """
    # NaN gelen ya da bos string durumunda bos don, yoksa split hata verir
    if not isinstance(metin, str) or not metin.strip():
        return ""

    # virgule gore parcala, her notanin etrafindaki boslugu temizle, kucult
    notalar = [n.strip().lower() for n in metin.split(",")]

    # ayni notanin birden fazla kez gecmesini onlemek icin set + liste kullandim
    # set kullanmak sirayi bozuyor, bu yuzden goruldu listesi tutuyorum
    goruldu, benzersiz = set(), []
    for n in notalar:
        if n and n not in goruldu:
            goruldu.add(n)
            benzersiz.append(n)

    # TF-IDF boslukla ayrilan kelimeleri isliyor, virgul degil
    return " ".join(benzersiz)


# --------------------------------------------------------------
# load_data
# --------------------------------------------------------------
def load_data(csv_yolu: str = VARSAYILAN_CSV) -> pd.DataFrame:
    """
    parfumes.csv dosyasini okur, her koku ailesinden 300 ornek alir,
    notalari normalize eder ve model icin hazir hale getirir.

    Parametreler
    ------------
    csv_yolu : str
        CSV dosyasinin yolu (varsayilan: data/parfumes.csv)

    Dondurur
    --------
    pd.DataFrame
        Temizlenmis ve filtrelenmis parfum verisi.
        Yeni sutun: tum_notalar (ust + orta + alt notalarin birlesimi)

    Hatalar
    -------
    FileNotFoundError : dosya bulunamazsa
    ValueError        : zorunlu sutunlar eksikse
    """

    # 1. Dosyanin gercekten var olup olmadigini kontrol ediyoruz
    # yoksa anlasilir bir hata mesaji veriyoruz
    if not os.path.exists(csv_yolu):
        raise FileNotFoundError(
            f"Dosya bulunamadi: {csv_yolu}\n"
            "data/ klasorune parfumes.csv ekleyin."
        )

    # 2. CSV'yi oku - encoding belirtmek onemli, bazi sistemlerde hata verebiliyor
    df = pd.read_csv(csv_yolu, encoding="utf-8")

    # 3. Ingilizce sutun adlarini Turkce'ye cevir (sadece var olanlari)
    df = df.rename(columns={k: v for k, v in SUTUN_ESLEME.items() if k in df.columns})

    # 4. Modelin calismasi icin kesinlikle olmasi gereken sutunlari kontrol et
    zorunlu = ["marka", "ad", "koku_ailesi", "ust_nota", "orta_nota", "alt_nota"]
    eksik = [s for s in zorunlu if s not in df.columns]
    if eksik:
        raise ValueError(f"CSV'de eksik sutunlar: {eksik}")

    # 5. Nota sutunlarindaki bos degerler string islemlerinde hata verir,
    # onlari bos string ile dolduruyoruz
    for s in ["ust_nota", "orta_nota", "alt_nota"]:
        df[s] = df[s].fillna("")

    # 6. Uc notasi da tamamen bos olan satirlarin modele katki yapamayacagi icin cikariyoruz
    df = df[
        df["ust_nota"].str.strip().ne("") |
        df["orta_nota"].str.strip().ne("") |
        df["alt_nota"].str.strip().ne("")
    ].copy()

    # 7. Veri setinde bazi koku aileleri cok daha fazla temsil ediliyor
    # bunu onlemek icin her aileden esit sayida kayit aliyoruz (dengeli ornekleme)
    parcalar = []
    for aile, grup in df.groupby("koku_ailesi"):
        # gruptaki kayit sayisi 300'den azsa hepsini al, fazlaysa 300 tane sec
        ornek = grup.sample(n=min(AILE_BASI_KAYIT, len(grup)), random_state=42)
        parcalar.append(ornek)
    df = pd.concat(parcalar).reset_index(drop=True)
    print(f"[veri] Ornekleme: {len(df)} kayit | {df['koku_ailesi'].nunique()} koku ailesi")

    # 8. Tum nota sutunlarini normalize et (kucuk harf, tekrarsiz, boslukla ayirma)
    for s in ["ust_nota", "orta_nota", "alt_nota"]:
        df[s] = df[s].apply(_normalize_notalar)

    # 9. Model tek bir metin sutunu uzerinden calisacak
    # ust + orta + alt notalari birlesik bir sutunda topluyoruz
    df["tum_notalar"] = (
        df["ust_nota"] + " " + df["orta_nota"] + " " + df["alt_nota"]
    ).str.strip().str.replace(r"\s+", " ", regex=True)

    # 10. Bazi veri setlerinde parfum_id sutunu olmayabiliyor
    # yoksa P0001, P0002... seklinde otomatik olusturuyoruz
    if "parfum_id" not in df.columns:
        df.insert(0, "parfum_id", [f"P{i+1:04d}" for i in range(len(df))])

    # 11. Puan sutunu da olmayabilir, varsayilan olarak 4.0 atiyoruz
    if "ortalama_puan" not in df.columns:
        df["ortalama_puan"] = 4.0

    print(f"[veri] {len(df)} parfum yuklendi -> {csv_yolu}")
    return df


# --------------------------------------------------------------
# build_tfidf
# --------------------------------------------------------------
def build_tfidf(df: pd.DataFrame):
    """
    tum_notalar sutunundan TF-IDF matrisi olusturur.

    Parametreler
    ------------
    df : pd.DataFrame
        load_data() ciktisi

    Dondurur
    --------
    vectorizer    : TfidfVectorizer  (yeni sorgular icin kullanilir)
    tfidf_matrisi : scipy sparse matrix  (n_parfum x n_kelime)

    Hatalar
    -------
    ValueError : tum_notalar sutunu eksikse
    """
    # load_data() atlanip dogrudan bu fonksiyon cagirilirsa hata versin
    if "tum_notalar" not in df.columns:
        raise ValueError(
            "tum_notalar sutunu bulunamadi.\n"
            "Once load_data() fonksiyonunu calistirin."
        )

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",   # boslukla ayrilan her token bir kelime sayilsin
        ngram_range=(1, 2),        # hem tek kelime hem de ikili kombinasyonlari yakala
        sublinear_tf=True,         # cok tekrar eden notalarin agirligini dengelemek icin log olcegi
        min_df=1,                  # en az 1 belgede gectikce dahil et, cok kucuk veri seti oldugu icin 1 biraktim
    )

    # fit_transform hem vektorizetoru egitir hem de tum parfumlerin matrisini olusturur
    tfidf_matrisi = vectorizer.fit_transform(df["tum_notalar"])
    print(
        f"[veri] TF-IDF matrisi: "
        f"{tfidf_matrisi.shape[0]} parfum x {tfidf_matrisi.shape[1]} ozellik"
    )
    return vectorizer, tfidf_matrisi


# --------------------------------------------------------------
# Hizli test
# --------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    print("\n--- Ornek Veriler (ilk 3 satir) ---")
    print(df[["ad", "marka", "koku_ailesi", "ust_nota", "orta_nota", "alt_nota"]].head(3).to_string())

    print("\n--- Koku Ailesi Dagilimi ---")
    print(df["koku_ailesi"].value_counts())

    print("\n--- TF-IDF ---")
    vec, mat = build_tfidf(df)
    print(f"Matris boyutu: {mat.shape}")