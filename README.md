# PerfumeAI: Koku Notası Tabanlı Parfüm Öneri Sistemi

**PerfumeAI**, kullanıcıların koku tercihlerini (üst, orta ve alt notalar) analiz ederek, binlerce parfüm arasından en uygun seçenekleri belirleyen ve geçmiş tercihlerine göre kişiselleştirilmiş profil oluşturan bir Python uygulamasıdır.

| Özellik | Değer |
| --- | --- |
| **Proje Adı** | PerfumeAI – Akıllı Öneri Sistemi |
| **Veri Kaynağı** | HuggingFace (pelegelraz/perfumes-dataset) |
| **Veri Hacmi** | 3.000 Dengeli Kayıt (10.000 Ham Veri) |
| **Algoritma** | TF-IDF + Cosine Similarity |
| **Arayüz** | Streamlit |

---

##  1. Proje Özeti

Parfüm dünyasındaki "kör alış" (koklamadan satın alma) riskini minimize etmeyi hedefleyen bu proje, içerik tabanlı filtreleme yöntemini kullanarak kullanıcının sevdiği notaları matematiksel bir "Zevk Vektörü"ne dönüştürür.

###  Temel Hedefler

* **Hassas Eşleştirme:** Kullanıcın seçtiği notalarla en yüksek kosinüs benzerliğine sahip parfümleri bulmak.
* **Dengeli Öneri:** Her koku ailesinden (Woody, Floral, Citrus vb.) eşit oranda beslenen bir veri setiyle tarafsız sonuçlar sunmak.
* **Kişiselleştirilmiş Geçmiş:** Beğenilen parfümleri `JSON` formatında saklayarak bir sonraki oturumda otomatik profil önerisi yapmak.

---

##  2. Sistem Mimarisi ve Veri Hattı (Data Pipeline)

Proje, birbirine sıkı sıkıya bağlı üç ana modülden oluşur:

### 2.1. Veri Akışı

1. **Veri Çekme:** HuggingFace üzerinden 10.000 parfüm kaydı çekilir.
2. **Normalizasyon:** Notalar küçük harfe çevrilir, boşluklar temizlenir ve benzersiz koku imzaları oluşturulur.
3. **Dengeli Örnekleme:** 10 ana koku ailesinden 300'er kayıt seçilerek **3.000 kayıtlık** optimize edilmiş alt küme oluşturulur.
4. **Vektörizasyon:** `TfidfVectorizer` ile 1.814 farklı öznitelik (nota) çıkarılarak matris oluşturulur.

### 2.2. Dizin Yapısı

```text
perfumeai/
├── data/
│   ├── parfumes.csv          # 3.000 kayıtlık temizlenmiş dengeli veri seti
│   └── kullanici_gecmis.json # Kullanıcının beğendiği parfümler (JSON)
├── src/
│   ├── veri.py               # Veri temizleme, Normalizasyon, TF-IDF üretimi
│   ├── model.py              # Kosinüs benzerliği ve Profil çıkarma motoru
│   └── arayuz.py             # Streamlit Web Arayüzü
├── requirements.txt
└── README.md

```

---

## 3. Kullanılan Algoritmalar

### 3.1. TF-IDF Vektörizasyonu

Notaların sadece varlığına değil, ayırt ediciliğine odaklanılır.

* **Analyzer:** `word` (Kelime bazlı)
* **N-gram:** `(1, 2)` (Örn: "beyaz misk" ikili bir yapı olarak algılanır)
* **Sublinear TF:** Logaritmik ölçekleme ile nadir notaların ağırlığı artırılır.

### 3.2. Kosinüs Benzerliği

Kullanıcı sorgusu ($A$) ve parfüm vektörü ($B$) arasındaki açı hesaplanır. Skor **1.0**'a ne kadar yakınsa, parfüm kullanıcının zevkine o kadar uygundur.

---

##  4. Ekip ve Görev Dağılımı

| Kişi | Rol | Sorumluluklar |
| --- | --- | --- |
| **Kişi 1** | **Veri Mühendisi** | HuggingFace entegrasyonu, Dengeli örnekleme (Balanced Sampling), `load_data()` ve `build_tfidf()` fonksiyonları. |
| **Kişi 2** | **Model Mühendisi** | `oneri_yap()` ve `profil_cikar()` motorları, benzerlik matris hesaplamaları ve birim testler. |
| **Kişi 3** | **UI Geliştirici** | Streamlit arayüzü, çoklu nota seçimi, ürün kartları ve JSON veri saklama süreçleri. |

---

##  5. Kurulum ve Çalıştırma

### 5.1. Gereksinimler

* Python 3.10+
* `pip` paket yöneticisi

### 5.2. Kurulum

```bash
git clone https://github.com/murtaDuElama/perfumeAI.git
cd perfumeAI
pip install -r requirements.txt

```

### 5.3. Uygulamayı Başlatma

```bash
streamlit run src/arayuz.py

```

---

## 6. Test Sonuçları

Veri modülü Google Colab ortamında test edilmiş ve aşağıdaki çıktılar doğrulanmıştır:

* **Örnekleme:** 10 koku ailesi x 300 kayıt = 3.000 toplam veri.
* **Matris Boyutu:** 3.000 Parfüm x 1.814 Özellik.
* **Hız:** <100ms sorgu süresi.

