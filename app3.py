import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from difflib import SequenceMatcher
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentence_transformers import SentenceTransformer, util

# Load model BERT dan spaCy NER
@st.cache_resource
def load_models():
    bert_model = SentenceTransformer("indobenchmark/indobert-base-p1")
    nlp = spacy.load("xx_ent_wiki_sm")  
    return bert_model, nlp

bert_model, nlp = load_models()

# Custom political parties list for Indonesian context
POLITICAL_PARTIES = {
    "pkb", "gerindra", "golkar", "pdip", "pdi-p", "demokrat", "pan", "pks", 
    "nasdem", "ppp", "hanura", "psi", "perindo", "pkpi", "berkarya"
}

# Normalisasi teks
def normalize_text(text):
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# Ekstrak entitas dengan penekanan pada partai politik
def extract_entities(text):
    doc = nlp(text)
    # Extract standard named entities
    entities = set(ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE'])
    
    # Add custom detection for Indonesian political parties
    text_lower = text.lower()
    for party in POLITICAL_PARTIES:
        if party in text_lower.split():
            entities.add(party)
    
    return entities

# Extract political parties specifically
def extract_political_parties(text):
    text_lower = text.lower()
    found_parties = set()
    for party in POLITICAL_PARTIES:
        if party in text_lower.split():
            found_parties.add(party)
    return found_parties

# Load dan proses dataset
@st.cache_data
def load_data():
    df = pd.read_excel("dataset_cnn_summarized - Copy.xlsx")
    df['text_norm'] = df['title'].apply(normalize_text)
    df['entities'] = df['title'].apply(lambda x: extract_entities(str(x)))
    df['political_parties'] = df['title'].apply(extract_political_parties)
    bert_embeddings = bert_model.encode(df['text_norm'].tolist(), convert_to_tensor=True)
    return df, bert_embeddings

df, bert_embeddings = load_data()

# Cek perubahan kecil tapi mencurigakan
def is_suspicious_change(input_text, matched_text, threshold_word_diff=0.2):
    input_words = input_text.lower().split()
    matched_words = matched_text.lower().split()
    matcher = SequenceMatcher(None, input_words, matched_words)
    ratio = matcher.ratio()

    # Hitung proporsi perbedaan kata
    num_diff = sum(1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != 'equal')
    total = max(len(input_words), len(matched_words))
    prop_diff = num_diff / total

    # Jika kalimat sangat mirip tapi hanya beda sedikit, itu mencurigakan
    if ratio > 0.85 and prop_diff < threshold_word_diff:
        return True, prop_diff
    return False, prop_diff

# Prediksi
def prediksi_berita(judul_berita, threshold_valid=0.7, threshold_rekom=0.35):
    judul_norm = normalize_text(judul_berita)
    input_entities = extract_entities(judul_berita)
    input_parties = extract_political_parties(judul_berita)
    input_vec = bert_model.encode(judul_norm, convert_to_tensor=True)

    # Hitung similarity
    similarity = util.pytorch_cos_sim(input_vec, bert_embeddings)[0].cpu().numpy()
    idx_max = np.argmax(similarity)
    max_score = similarity[idx_max]
    berita_mirip = df.iloc[idx_max]

    st.write(f"🔍 Similarity tertinggi: {max_score:.4f}")
    st.write(f"- Judul Mirip: {berita_mirip['title']}")

    entities_mirip = berita_mirip['entities']
    parties_mirip = berita_mirip.get('political_parties', set())

    # Check for political party mismatch specifically (tanpa output detail)
    party_mismatch = False
    if input_parties != parties_mirip and (input_parties or parties_mirip):
        party_mismatch = True

    # Deteksi perubahan kecil tapi penting
    suspicious, prop_diff = is_suspicious_change(judul_berita, berita_mirip['title'])

    # Modified validation logic with stricter entity checking
    if (max_score >= threshold_valid and 
        input_entities == entities_mirip and 
        not party_mismatch):
        if suspicious:
            st.warning(f"⚠️ Terlalu mirip tapi beda di {prop_diff*100:.1f}% bagian kalimat")
            st.error("❌ Deteksi: HOAX (Modifikasi kecil tapi mencurigakan)")
            st.write("🔎 Berdasarkan berita yang paling mirip:")
            st.markdown(f"- **Judul Asli**: {berita_mirip['title']}")
            st.markdown(f"- 🌐 **Link**: {berita_mirip.get('url', 'Tidak ada URL')}")
        else:
            st.success("✅ Deteksi: VALID (Mirip dan entitas cocok)")
            st.write(f"- URL  : {berita_mirip.get('url', 'Tidak ada URL')}")
            try:
                tahun = pd.to_datetime(berita_mirip.get('date', None)).year
                st.write(f"📅 Tahun Terbit: {tahun}")
            except:
                st.warning("⚠️ Tahun tidak bisa diambil.")
    else:
        # Provide more specific reason for HOAX classification
        reasons = []
        if max_score < threshold_valid:
            reasons.append("similarity rendah")
        if input_entities != entities_mirip or party_mismatch:
            reasons.append("entitas berbeda")
            
        reason_str = " dan ".join(reasons)
        st.error(f"❌ Deteksi: HOAX ({reason_str})")

        # Rekomendasi berita valid
        df_valid = df[df['label'] == 0].copy()
        if not df_valid.empty:
            valid_embeddings = bert_model.encode(df_valid['text_norm'].tolist(), convert_to_tensor=True)
            sim_valid = util.pytorch_cos_sim(input_vec, valid_embeddings)[0].cpu().numpy()
            max_sim_rekom = np.max(sim_valid)

            if max_sim_rekom >= threshold_rekom:
                idx_valid = np.argmax(sim_valid)
                rekom = df_valid.iloc[idx_valid]
                st.write("🔁 Rekomendasi Berita Valid:")
                st.write(f"- Judul: {rekom['title']}")
                st.write(f"- URL  : {rekom.get('url', 'Tidak ada URL')}")
            else:
                st.warning("⚠️ Tidak ada rekomendasi valid yang cukup mirip.")
        else:
            st.warning("⚠️ Tidak ada berita valid dalam dataset.")

# Antarmuka Streamlit
st.title("🕵️‍♂️ Deteksi Berita Hoax dengan NLP + BERT")

judul_input = st.text_input("Masukkan Judul Berita:")
if st.button("Deteksi"):
    if judul_input.strip() == "":
        st.warning("Masukkan judul berita terlebih dahulu.")
    else:
        prediksi_berita(judul_input)