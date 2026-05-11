"""
app.py  –  PerfumeAI  |  Kişi 3 (Lüks & Minimalist Tasarım)
Çalıştır: streamlit run src/app.py
"""

import sys, os
# app.py src/ klasörünün içinde, veri.py ve model.py'yi import edebilmek için
# bulunduğu dizini Python'un arama yoluna ekliyoruz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from veri import load_data, build_tfidf
from model import recommend

# ── Sayfa yapılandırması ──────────────────────────────────────
# layout="wide" ile kenar boşluklarını azaltıp daha geniş bir görünüm elde ettik
st.set_page_config(page_title="PerfumeAI | Boutique", page_icon="✨", layout="wide")

# ── Tema & Stil (Boutique Lüks Tasarım) ───────────────────────
# Tüm CSS'i burada tanımladık, Streamlit'in varsayılan stillerini geçersiz kılıyoruz
# !important kullanmak zorunda kaldık çünkü Streamlit kendi stillerini sonradan yüklüyor
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');

* { font-family: 'Montserrat', sans-serif; box-sizing: border-box; }

/* Başlıklarda Playfair Display kullanıyoruz, daha lüks bir görünüm veriyor */
h1, h2, h3, h4, h5, h6, .sayfa-baslik h1, .kart-ad, .panel-baslik, [data-testid="stMetricValue"] { 
    font-family: 'Playfair Display', serif !important; 
}

/* Genel zemin (İnci/Krem Beyazı) */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #FDFBF7;
}

/* Global metin */
html, body, [class*="css"], .stMarkdown, p, li, span, div {
    color: #4A443D;
}

/* ── Üst bar ── */
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #F4F0EA !important;
    border-right: 1px solid #E8E2D9;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * { color: #4A443D !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h4 { color: #2C2825 !important; font-family: 'Playfair Display', serif !important; letter-spacing: 0.5px;}

/* ── Input ── */
.stTextInput input, .stTextInput textarea {
    background: #FFFFFF !important;
    color: #2C2825 !important;
    border: 1px solid #E8E2D9 !important;
    border-radius: 4px !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.02) !important;
}
.stTextInput input:focus {
    border-color: #B8997A !important;
    box-shadow: 0 0 0 1px #B8997A !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div,
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #E8E2D9 !important;
    border-radius: 4px !important;
    color: #4A443D !important;
}
[data-baseweb="popover"] { background: #FFFFFF !important; border: 1px solid #E8E2D9 !important; }
[role="option"]          { background: #FFFFFF !important; color: #4A443D !important; }
[role="option"]:hover    { background: #F4F0EA !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #B8997A !important;
}

/* ── Buton ── */
.stButton > button {
    width: 100%;
    background: #B8997A !important;   /* altın kahve ton - tema rengi */
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { 
    background: #9C8268 !important;   /* hover'da biraz daha koyu ton */
    color: #FFFFFF !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(156,130,104,0.25);
}

/* ── Metrikler ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E2D9;
    border-radius: 8px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
[data-testid="stMetricLabel"] p { color: #8A8175 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 1.5px; }
[data-testid="stMetricValue"]   { color: #2C2825 !important; font-size: 2rem !important; font-weight: 400; }

/* ── Ayraç ── */
hr { border: none; border-top: 1px solid #E8E2D9 !important; margin: 1.5rem 0; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E8E2D9 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
[data-testid="stExpander"] summary { color: #4A443D !important; font-size: 0.9rem !important; font-weight: 500; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #E8E2D9; border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] th { background: #F4F0EA !important; color: #5C5449 !important;
    font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;}
[data-testid="stDataFrame"] td { color: #4A443D !important; font-size: 0.88rem !important; }

/* ── Özel kartlar ── */
/* her parfüm sonucu bu kart yapısıyla gösteriliyor */
.kart {
    background: #FFFFFF;
    border: 1px solid #E8E2D9;
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    position: relative;
}
.kart:hover { border-color: #B8997A; box-shadow: 0 6px 20px rgba(184,153,122,0.08); }

.kart-no    { position: absolute; top: 1.8rem; right: 2rem; font-size: 0.8rem; color: #D1C8BC; font-weight: 400; font-family: 'Playfair Display', serif; font-style: italic;}
.kart-ad    { font-size: 1.5rem; color: #2C2825; margin-bottom: 0.3rem; font-weight: 600; letter-spacing: 0.5px;}
.kart-marka { font-size: 0.85rem; color: #8A8175; margin-bottom: 1.2rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 500; }

.tag { display: inline-block; padding: 4px 12px; border-radius: 20px;
       font-size: 0.65rem; font-weight: 600; margin-right: 8px; text-transform: uppercase; letter-spacing: 1px; }
.tag-aile { background: #F4F0EA; color: #5C5449; }
.tag-tip  { background: #FDF6E3; color: #8C7A6B; }
.tag-skor { background: #F9F1F2; color: #9A7B7F; border: 1px solid #F0DFE1; }

/* nota bloğunu 3 sütuna böldük: üst / orta / alt nota yan yana görünsün */
.nota-blok { margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #F4F0EA;
             display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
.nota-label { font-size: 0.65rem; font-weight: 600; color: #B8997A;
              text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
.nota-val   { font-size: 0.9rem; color: #4A443D; line-height: 1.6; }

/* ── Bilgi paneli ── */
.panel {
    background: #FFFFFF;
    border: 1px solid #E8E2D9;
    border-radius: 8px;
    padding: 2rem;
    height: 100%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.panel-baslik { font-size: 1.2rem; font-weight: 600; color: #2C2825;
                margin-bottom: 1rem; border-bottom: 1px solid #F4F0EA; padding-bottom: 0.8rem;}
.panel-metin  { font-size: 0.9rem; color: #665F56; line-height: 1.7; }

/* ── Başlık alanı ── */
.sayfa-baslik { padding: 3rem 0 2.5rem; border-bottom: 1px solid #E8E2D9; margin-bottom: 2.5rem; text-align: center; }
.sayfa-baslik h1 { font-size: 2.8rem; font-weight: 600; color: #2C2825; margin: 0; letter-spacing: 1px; }
.sayfa-baslik span { color: #B8997A; font-style: italic; font-weight: 400; }
.sayfa-baslik p { font-size: 0.85rem; color: #8A8175; margin: 0.8rem 0 0; text-transform: uppercase; letter-spacing: 2px; }

/* ── Sonuç başlığı ── */
.sonuc-baslik { font-size: 0.9rem; color: #8A8175; margin-bottom: 2rem; text-align: center;
                padding-bottom: 1.5rem; border-bottom: 1px solid #E8E2D9; }
.sonuc-baslik strong { color: #2C2825; font-family: 'Playfair Display', serif; font-size: 1.2rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── Veri yükleme ──────────────────────────────────────────────
# @st.cache_resource ile veri sadece bir kez yükleniyor
# kullanıcı her arama yaptığında CSV'yi ve TF-IDF matrisini sıfırdan hesaplamak çok yavaş olurdu
@st.cache_resource(show_spinner="Koleksiyon hazırlanıyor...")
def veri_yukle():
    df = load_data()
    vec, mat = build_tfidf(df)
    return df, vec, mat

df, vectorizer, tfidf_matrix = veri_yukle()

# ── Session state ─────────────────────────────────────────────
# Streamlit her buton tıklamasında sayfayı yeniden render ediyor
# bu yüzden arama sonuçlarını session_state'e kaydediyoruz, yoksa sonuçlar kayboluyor
for k, v in [("sonuclar", None), ("arama_yapildi", False), ("son_sorgu", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────
# arama formu ve filtreler sol panelde, sonuçlar ana alanda gösteriliyor
with st.sidebar:
    st.markdown("#### Koku Profiliniz")
    sorgu = st.text_input(
        "Koku notaları",
        placeholder="Örn: bergamot cedar musk",
        label_visibility="collapsed",
    )
    # kullanıcıya sorgularını İngilizce yazmaları gerektiğini hatırlatıyoruz
    # veri seti İngilizce olduğu için Türkçe yazılırsa eşleşme bulunamıyor
    st.caption("Aramak istediğiniz notaları İngilizce ve boşlukla ayırarak girin.")

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
    st.markdown("#### Filtreler")

    # selectbox seçeneklerini veri setinden dinamik olarak çekiyoruz
    # böylece yeni bir koku ailesi eklenince kod değiştirmemize gerek kalmıyor
    secili_aile  = st.selectbox("Koku Ailesi", ["Tümü"] + sorted(df["koku_ailesi"].unique()))
    secili_marka = st.selectbox("Tasarımcı / Marka", ["Tümü"] + sorted(df["marka"].unique()))
    secili_tip   = st.selectbox("Parfüm Tipi", ["Tümü"] + sorted(df["tip"].unique()))

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    top_n = st.slider("Listelenecek Sonuç", 1, 10, 5)

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
    ara = st.button("Koleksiyonu Keşfet", use_container_width=True)

    if ara:
        if not sorgu.strip():
            st.warning("Lütfen en az bir nota girin.")
        else:
            # "Tümü" seçiliyse model.py'ye None gönderiyoruz, filtre uygulanmasın diye
            fa = None if secili_aile  == "Tümü" else secili_aile
            fm = None if secili_marka == "Tümü" else secili_marka
            ft = None if secili_tip   == "Tümü" else secili_tip

            # sonuçları session_state'e kaydediyoruz, sayfa yenilenince de kalıcı olsun
            st.session_state.sonuclar      = recommend(sorgu, df, vectorizer, tfidf_matrix,
                                                       top_n=top_n, koku_ailesi=fa, marka=fm, tip=ft)
            st.session_state.arama_yapildi = True
            st.session_state.son_sorgu     = sorgu  # sonuç başlığında göstermek için saklıyoruz

# ── Ana alan — Başlık ─────────────────────────────────────────
st.markdown("""
<div class="sayfa-baslik">
    <h1>Perfume<span>AI</span></h1>
    <p>Kişiselleştirilmiş Koku Eşleştirme Sistemi</p>
</div>
""", unsafe_allow_html=True)

# ── Ana alan — İçerik ─────────────────────────────────────────
if not st.session_state.arama_yapildi:
    # ilk açılışta veri seti özeti ve kullanım adımlarını gösteriyoruz
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parfüm", f"{len(df):,}")
    col2.metric("Marka",  df["marka"].nunique())
    col3.metric("Koku Ailesi", df["koku_ailesi"].nunique())
    col4.metric("Karakter (TF-IDF)", "1.814")

    st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="panel">
            <div class="panel-baslik">Adım I. İlham Verin</div>
            <div class="panel-metin">Sevdiğiniz koku notalarını soldaki alana İngilizce olarak girerek arzuladığınız parfüm profilini oluşturun.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="panel">
            <div class="panel-baslik">Adım II. Rafine Edin</div>
            <div class="panel-metin">Özel zevklerinize göre koku ailesi, favori tasarımcınız veya parfüm tipini seçerek sonuçları daraltın.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="panel">
            <div class="panel-baslik">Adım III. Keşfedin</div>
            <div class="panel-metin">Gelişmiş algoritmamız, girdiğiniz notalarla en uyumlu lüks parfümleri kosinüs benzerliği ile size sunsun.</div>
        </div>""", unsafe_allow_html=True)

else:
    sonuclar = st.session_state.sonuclar

    # filtrelere uyan parfüm yoksa ya da tüm benzerlik skorları 0 ise boş gelir
    if sonuclar is None or sonuclar.empty:
        st.error("Eşleşen parfüm bulunamadı. Lütfen filtrelerinizi esnetin veya farklı notalar deneyin.")
    else:
        # hangi sorgu için kaç sonuç bulunduğunu başlıkta gösteriyoruz
        st.markdown(
            f'<div class="sonuc-baslik"><strong>"{st.session_state.son_sorgu}"</strong> '
            f'esansı için size özel {len(sonuclar)} öneri hazırlandı.</div>',
            unsafe_allow_html=True
        )

        # her parfümü ayrı bir kart olarak render ediyoruz
        # HTML kullanmak zorunda kaldık çünkü Streamlit'in yerleşik kartları bu tasarıma uymuyor
        for i, row in sonuclar.iterrows():
            skor = round(row["benzerlik_skoru"], 4)
            st.markdown(f"""
            <div class="kart">
                <div class="kart-no">No. {i+1}</div>
                <div class="kart-ad">{row['ad']}</div>
                <div class="kart-marka">{row['marka']}</div>
                <span class="tag tag-aile">{row['koku_ailesi']}</span>
                <span class="tag tag-tip">{row['tip']}</span>
                <span class="tag tag-skor">Eşleşme: {skor}</span>
                <div class="nota-blok">
                    <div>
                        <div class="nota-label">Üst Nota</div>
                        <div class="nota-val">{row['ust_nota'] or '—'}</div>
                    </div>
                    <div>
                        <div class="nota-label">Orta Nota</div>
                        <div class="nota-val">{row['orta_nota'] or '—'}</div>
                    </div>
                    <div>
                        <div class="nota-label">Alt Nota</div>
                        <div class="nota-val">{row['alt_nota'] or '—'}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

        # tablo görünümü opsiyonel, varsayılan olarak kapalı
        with st.expander("Detaylı Görünüm"):
            tablo = sonuclar[["marka","ad","koku_ailesi","tip","benzerlik_skoru"]].copy()
            tablo.columns = ["Marka","Parfüm","Koku Ailesi","Tip","Uyum"]
            tablo["Uyum"] = tablo["Uyum"].apply(lambda x: f"{x:.4f}")
            st.dataframe(tablo, use_container_width=True, hide_index=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # yeni arama yapılınca session_state sıfırlanıyor ve karşılama ekranına dönüyor
        if st.button("Yeni Arama", use_container_width=False):
            st.session_state.arama_yapildi = False
            st.session_state.sonuclar = None
            st.rerun()