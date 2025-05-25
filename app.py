import streamlit as st
import pandas as pd
import numpy as np
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dan proses data sekali saja
@st.cache_data
def load_and_process_data():
    df = pd.read_excel("dataset_cnn_summarized.xlsx")

    # Setup normalizer
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    def normalize_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [stemmer.stem(w) for w in tokens]
        return ' '.join(tokens)

    # Normalisasi kolom judul
    df['text_norm'] = df['title'].apply(normalize_text)

    # Fit TF-IDF sekali saja
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['text_norm'])

    return df, tfidf, tfidf_matrix

df, tfidf, tfidf_matrix = load_and_process_data()

def normalize_text(text):
    # Ini fungsi normalisasi yang sama untuk input user
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def prediksi_berita(judul_berita, threshold_valid=0.5, threshold_rekom=0.35):
    judul_input = normalize_text(judul_berita)

    # cek exact match
    exact_match = df[df['text_norm'] == judul_input]
    if not exact_match.empty:
        berita = exact_match.iloc[0]
        st.success("✅ Deteksi: VALID (Exact Match)")
        st.write(f"- Judul: {berita['title']}")
        st.write(f"- URL  : {berita.get('url', 'Tidak ada URL')}")
        try:
            tahun = pd.to_datetime(berita.get('date', None)).year
            st.write(f"📅 Tahun Terbit: {tahun}")
        except:
            st.warning("⚠️ Tahun tidak bisa diambil.")
        return

    input_vec = tfidf.transform([judul_input])
    similarity = (input_vec @ tfidf_matrix.T).toarray()[0]
    idx_max = np.argmax(similarity)
    max_score = similarity[idx_max]
    berita_mirip = df.iloc[idx_max]

    st.write(f"Similarity tertinggi dengan data: {max_score:.4f}")
    if max_score >= threshold_valid:
        st.success("✅ Deteksi: VALID (Similaritas Tinggi)")
        st.write(f"- Judul: {berita_mirip['title']}")
        st.write(f"- URL  : {berita_mirip.get('url', 'Tidak ada URL')}")
        try:
            tahun = pd.to_datetime(berita_mirip.get('date', None)).year
            st.write(f"📅 Tahun Terbit: {tahun}")
        except:
            st.warning("⚠️ Tahun tidak bisa diambil.")
    else:
        st.error("❌ Deteksi: HOAX")

        df_valid = df[df['label'] == 0].copy()
        if df_valid.empty:
            st.warning("🔁 Tidak ada berita valid untuk direkomendasikan.")
            return

        valid_vecs = tfidf.transform(df_valid['text_norm'])
        similarity_valid = (input_vec @ valid_vecs.T).toarray()[0]

        max_sim_rekom = np.max(similarity_valid) if len(similarity_valid) > 0 else 0
        if max_sim_rekom < threshold_rekom:
            st.warning("🔁 Tidak ada rekomendasi berita valid yang cukup mirip.")
            return

        idx_valid = df_valid.index[np.argmax(similarity_valid)]
        rekom = df.loc[idx_valid]
        st.write("🔁 Rekomendasi Berita Valid:")
        st.write(f"- Judul: {rekom['title']}")
        st.write(f"- URL  : {rekom.get('url', 'Tidak ada URL')}")

st.title("Deteksi Berita Hoax - Masukkan Judul Berita")

judul_user = st.text_input("Masukkan Judul Berita:")

if st.button("Deteksi"):
    if judul_user.strip() == "":
        st.warning("Masukkan judul berita terlebih dahulu.")
    else:
        prediksi_berita(judul_user)
