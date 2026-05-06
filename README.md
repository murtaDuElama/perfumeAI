# perfumeAI
a perfume application
# PerfumeAI
## Koku Notası Tabanlı Parfüm Öneri Sistemi

| Özellik | Değer |
|---------|-------|
| **Proje Adı** | PerfumeAI – Koku Notası Tabanlı Parfüm Öneri Sistemi |
| **Proje Dili** | Python 3.10+ |
| **Geliştirme Ortamı** | Visual Studio Code |
| **Ekip** | 3 Kişi |
| **Zorluk Seviyesi** | Orta Seviye |
| **Versiyon** | v1.0.0 |

---

## 1. Proje Hakkında

PerfumeAI, kullanıcının sevdiği koku notalarını girerek bu notaları içeren parfümleri bulmak ve geçmiş kullanım tercihlerine göre kişiselleştirilmiş öneri sunmak amacıyla geliştirilmiş bir Python uygulamasıdır.

Proje, makine öğrenimi kavramlarını gerçek hayat problemine uygulayan, modüler yapıda, orta zorluk seviyesinde bir çalışmadır.

### 1.1. Problem Tanımı

- Parfüm alırken kokuyu önceden test etmek mümkün değil (kör alış sorunu).
- Binlerce ürün arasında kişisel nota tercihlerine uygun parfümü bulmak çok zaman alıyor.
- Daha önce beğendiği parfümler baz alarak yeni öneri yapan bir araç yok.

### 1.2. Proje Amacı

- Kullanıcının seçtiği koku notalarına göre en benzer parfümleri listelemek.
- Daha önce beğendiği parfümlerin notalarından otomatik profil çıkararak öneri yapmak.
- Beğendiği parfümleri JSON dosyasına kaydedip sonraki oturumda bu geçmişi kullanmak.

---

## 2. Sistem Mimarisi

Proje üç bağımsız modülden oluşur. Her modül tek bir sorumluluğa sahiptir ve diğer modüllerle standart fonksiyon çağrıları üzerinden iletişir.

- **Modül 1 – Veri**
- **Modül 2 – Model**
- **Modül 3 – Arayüz**

```
parfumes.csv → kullanici_gecmis.json → veri temizleme → nota vektörizasyon
                                                                   ↓
                                                TF-IDF Matrisi → Kosinüs Benzerliği
                                                                   ↓
                                          öneri_yap() fonksiyonu ← profil_cikar() fonksiyonu
                                                                   ↓
                                                        Streamlit Arayüzü
                                                        (nota seçimi, öneri kartları, beğeni kaydetme)
```

### 2.1. Veri Akış Diyagramı

1. Kullanıcı Streamlit arayüzünde sevdiği notaları seçer.
2. Seçilen notalar TF-IDF ile sayısal vektöre dönüştürülür.
3. Parfüm veri setindeki tüm parfümlerle kosinüs benzerliği hesaplanır.
4. En yüksek benzerlik skoruna sahip parfümler ekrana listelenir.
5. Kullanıcı beğendiği parfüm işaretlerse bu bilgi JSON dosyasına kaydedilir.
6. Sonraki oturumda JSON'daki geçmiş parfümlerin notasından otomatik profil çıkarılır ve öneri yapılır.

### 2.2. Dizin Yapısı

```
perfumeai/
├── data/
│   ├── parfumes.csv          # ~500-1000 parfüm kaydını içeren veri seti
│   └── kullanici_gecmis.json # kullanıcının beğendiği parfümler
├── src/
│   ├── veri.py              # CSV okuma, temizleme, TF-IDF matrisi
│   ├── model.py             # kosinüs benzerliği, öneri ve profil fonksiyonları
│   └── arayuz.py            # Streamlit uygulaması
├── requirements.txt
└── README.md
```

---

## 3. Kullanılacak Model ve Algoritma

Proje yalnızca içerik tabanlı filtreleme (content-based filtering) kullanır. Bu, orta seviye için hem yeterince öğretici hem de uygulanabilir bir yaklaşımdır.

### 3.1. İçerik Tabanlı Filtreleme

| | |
|---|---|
| **Algoritma** | TF-IDF Vektörizasyon + Kosinüs Benzerliği |
| **Kütüphane** | scikit-learn (TfidfVectorizer, cosine_similarity) |

**Çalışma Mantığı:**

1. Her parfümün koku notaları (üst nota, orta nota, alt nota) tek bir metin dizisi olarak birleştirilir.
2. TfidfVectorizer bu metinleri sayısal vektöre dönüştürür.
3. Kullanıcının girdisiyle (veya geçmiş profilden oluşan vektörle) tüm parfümler arasındaki kosinüs benzerliği hesaplanır.
4. Benzerlik skoru en yüksek parfümler sıralı şekilde öneri olarak sunulur.

### 3.2. Kullanıcı Profili Oluşturma

Kullanıcının daha önce beğendiği parfümlerin notaları toplanarak ortalama bir "zevk vektörü" hesaplanır. Bu vektör sonraki öneri sorgusunda kullanıcının tercihi olarak kullanılır.

```python
# Örnek mantık (gerçek kod değil, açıklayıcı)
begeni_notalar = [parfum['notalar'] for parfum in gecmis]
profil_metni   = ' '.join(begeni_notalar)
# Bu profil metni TF-IDF ile vektörleştirilip öneri için kullanılır
```

### 3.3. Kullanılacak Kütüphaneler

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| pandas | >=2.0 | CSV okuma ve veri manipülasyonu |
| scikit-learn | >=1.3 | TF-IDF ve kosinüs benzerliği |
| streamlit | >=1.30 | Web tabanlı kullanıcı arayüzü |
| json | Yerleşik | Kullanıcı geçmişi kaydetme/okuma |
| pytest | >=7.0 | Birim testleri |

---

## 4. Ekip Dağılımı

Her kişi bağımsız bir modül geliştirir. Modüller birbirini beklemeden paralel çalışabilir; entegrasyon son haftada yapılır.

| Kişi | Rol | Sorumluluklar |
|------|-----|---------------|
| Kişi 1 | Veri Müh. | parfumes.csv veri setini temin etme, temizleme, TF-IDF matrisini oluşturma (veri.py) |
| Kişi 2 | Model Müh. | Kosinüs benzerliği ile öneri fonksiyonu, kullanıcı profili çıkartma, birim testler (model.py) |
| Kişi 3 | UI Geliştirici | Streamlit arayüzü, nota seçim ekranı, öneri kartları, JSON kayıt/okuma (arayuz.py) |

### 4.1. Kişi 1 – Veri Modülü (veri.py)

**Görevler:**

- Kaggle'dan ~500-1000 parfümlük bir veri seti bulmak veya oluşturmak.
- Eksik değerleri temizlemek, nota sütunlarını normalize etmek.
- TF-IDF matrisini hesaplayıp kaydetmek.
- `load_data()` ve `build_tfidf()` fonksiyonlarını yazmak.

**Teslim Edilecekler:**

- `data/parfumes.csv`
- `src/veri.py` – `load_data()`, `build_tfidf()` fonksiyonları

### 4.2. Kişi 2 – Model Modülü (model.py)

**Görevler:**

- Kosinüs benzerliği ile en yakın parfümleri bulan `oneri_yap(notalar, n)` fonksiyonu.
- Geçmiş parfümlerden kullanıcı profili oluşturan `profil_cikar(gecmis)` fonksiyonu.
- Her iki fonksiyon için pytest ile test yazmak.

**Teslim Edilecekler:**

- `src/model.py` – `oneri_yap()`, `profil_cikar()` fonksiyonları
- `tests/test_model.py` – birim testleri

### 4.3. Kişi 3 – Arayüz Modülü (arayuz.py)

**Görevler:**

- Streamlit ile nota seçim ekranı oluşturmak (multiselect bileşeni).
- Öneri sonuçlarını kart formatında göstermek (marka, ad, notalar, skor).
- "Beğendim" butonuyla parfümü JSON'a kaydetmek.
- Sayfa başında JSON'dan geçmişi okuyarak profil önerisi göstermek.

**Teslim Edilecekler:**

- `src/arayuz.py` – Streamlit uygulaması
- `data/kullanici_gecmis.json` – boş şablon dosya

---

## 5. Veri Seti

- **Kaynak:** Kaggle – "Perfume Recommendation Dataset" veya benzer açık kaynak
- **Boyut:** 500 – 1000 parfüm kaydı (yeterli, yönetilebilir)

Her kayıtta bulunması gereken minimum alanlar:

- `parfum_id`, `ad`, `marka`
- `ust_nota`, `orta_nota`, `alt_nota` (virgülle ayrılmış liste)
- `koku_ailesi` (çiçeksi, odunsu, baharatlı, vb.)
- `ortalama_puan` (1-5 arası)

---

## 6. Kurulum ve Çalıştırma

### 6.1. Gereksinimler

- Python 3.10+
- pip

### 6.2. Kurulum

```bash
git clone https://github.com/kullanici/perfumeai.git
cd perfumeai
pip install -r requirements.txt
```

### 6.3. Uygulamayı Başlatma

```bash
streamlit run src/arayuz.py
```

### 6.4. Testleri Çalıştırma

```bash
pytest tests/
```
