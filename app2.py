import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

# Set page configuration
st.set_page_config(
    page_title="Deteksi Berita Hoax",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more attractive
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px #ccc;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .valid-result {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #4CAF50;
        margin: 10px 0 20px 0;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .hoax-result {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #f44336;
        margin: 10px 0 20px 0;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .warning-result {
        background-color: #fff8e1;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #FFC107;
        margin: 10px 0 20px 0;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .detail-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .metric-container {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: bold;
        color: #0277BD;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #01579B;
    }
    .big-result-icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model BERT dan spaCy NER
@st.cache_resource
def load_models():
    bert_model = SentenceTransformer("indobenchmark/indobert-base-p1")
    nlp = spacy.load("xx_ent_wiki_sm")  
    return bert_model, nlp

# Call load_models() once and assign to global variables
bert_model, nlp = load_models()

# Custom political parties list for Indonesian context
POLITICAL_PARTIES = {
    "pkb", "partai kebangkitan bangsa",
    "gerindra", "partai gerindra",
    "golkar", "partai golkar",
    "pdip", "pdi-p", "partai demokrasi indonesia perjuangan",
    "demokrat", "partai demokrat",
    "pan", "partai amanat nasional",
    "pks", "partai keadilan sejahtera",
    "nasdem", "partai nasdem",
    "ppp", "partai persatuan pembangunan",
    "hanura", "partai hati nurani rakyat",
    "psi", "partai solidaritas indonesia",
    "perindo", "partai persatuan indonesia",
    "pkpi", "partai keadilan dan persatuan indonesia",
    "berkarya", "partai berkarya"
}

# Enhanced synonym dictionary for better paraphrase handling
@st.cache_data
def build_enhanced_synonym_dict():
    """Build comprehensive synonym dictionary for Indonesian paraphrases"""
    base_synonyms = {
        "bakal": "akan", "bakalan": "akan", "hendak": "akan", "mau": "akan", 
        "akan": "akan", "mesti": "harus", "kudu": "harus", "wajib": "harus",
        "bilang": "kata", "ucap": "kata", "sebut": "kata", "tutur": "kata",
        "omong": "kata", "cerita": "kata", "ungkap": "kata", "sampaikan": "kata",
        "gak": "tidak", "ga": "tidak", "nggak": "tidak", "enggak": "tidak", 
        "tak": "tidak", "ndak": "tidak", "nda": "tidak", "bukan": "tidak",
        "lagi": "sedang", "udah": "sudah", "udahan": "sudah", "dah": "sudah", 
        "telah": "sudah", "pernah": "sudah", "sempat": "sudah",
        "esok": "besok", "kemarin": "kemarin", "tadi": "kemarin", "tempo": "lalu",
        "tapi": "tetapi", "namun": "tetapi", "akan tetapi": "tetapi", "cuma": "tetapi",
        "kecuali": "tetapi", "selain": "kecuali", "dan": "dan", "serta": "dan",
        "gimana": "bagaimana", "kenapa": "mengapa", "dimana": "di mana",
        "kayak": "seperti", "kaya": "seperti", "kayaknya": "sepertinya",
        "macam": "seperti", "mirip": "seperti", "ibarat": "seperti",
        "banget": "sangat", "sekali": "sangat", "amat": "sangat", "bener": "sangat",
        "parah": "sangat", "ekstrem": "sangat", "luar biasa": "sangat",
        "sama": "dengan", "ama": "dengan", "ma": "dengan", "bareng": "dengan",
        "bersama": "dengan", "beserta": "dengan",
        "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", 
        "pemkab": "pemerintah kabupaten", "presiden": "presiden",
        "menteri": "menteri", "gubernur": "gubernur", "bupati": "bupati",
        "walikota": "walikota", "camat": "camat", "lurah": "lurah",
        "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
        "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10",
        "puluhan": "banyak", "ratusan": "banyak", "ribuan": "banyak",
        "kabar": "berita", "info": "informasi", "laporan": "berita", "warta": "berita",
        "news": "berita", "breaking": "terbaru", "update": "terbaru", "terkini": "terbaru",
        "datang": "tiba", "pergi": "berangkat", "pulang": "kembali", "balik": "kembali",
        "cabut": "pergi", "nongol": "datang", "muncul": "datang", "hadir": "datang",
        "bagus": "baik", "jelek": "buruk", "gede": "besar", "kecil": "kecil",
        "mantap": "baik", "keren": "baik", "buruk": "buruk", "parah": "buruk",
        "oke": "baik", "ok": "baik", "fine": "baik",
        "doi": "dia", "nyokap": "ibu", "bokap": "ayah", "ortu": "orang tua",
        "gue": "saya", "gw": "saya", "ane": "saya", "lu": "kamu", "lo": "kamu",
        "ente": "kamu", "bro": "saudara", "sis": "saudara",
        "ngaku": "mengaku", "ngomong": "bicara", "ngasih": "memberi",
        "ngambil": "mengambil", "ngeliat": "melihat", "ngedenger": "mendengar",
        "nggih": "ya", "injih": "ya", "iya": "ya", "yoi": "ya", "yup": "ya",
        "enggeh": "ya", "oke": "ya", "siap": "ya",
        "gadget": "perangkat", "smartphone": "ponsel", "laptop": "komputer",
        "online": "daring", "offline": "luring", "update": "pembaruan",
        "duit": "uang", "perak": "uang", "cuan": "keuntungan", "untung": "keuntungan",
        "rugi": "kerugian", "bangkrut": "pailit", "sukses": "berhasil"
    }
    return base_synonyms

@st.cache_data 
def auto_generate_synonyms(_df):
    """Automatically detect synonym patterns from dataset with enhanced coverage"""
    try:
        auto_synonyms = build_enhanced_synonym_dict().copy()
        all_words = []
        word_pairs = []
        
        for title in _df['title'].dropna():
            words = str(title).lower().split()
            clean_words = [re.sub(r'[^\w]', '', w) for w in words if len(w) > 2]
            all_words.extend(clean_words)
            
            for i, word in enumerate(clean_words):
                for j, other_word in enumerate(clean_words):
                    if i != j and abs(i-j) <= 3:
                        word_pairs.append((word, other_word))
        
        # word_freq = pd.Series(all_words).value_counts()
        
        dataset_patterns = {
            "dikabarkan": "diberitakan", "dilaporkan": "diberitakan",
            "diklaim": "dinyatakan", "diungkap": "dikatakan",
            "terungkap": "diketahui", "terbongkar": "diketahui",
            "mencuat": "muncul", "merebak": "menyebar",
            "viral": "terkenal", "heboh": "ramai", "gaduh": "ramai",
            "kontroversi": "perdebatan", "polemik": "perdebatan",
            "somasi": "teguran", "gugatan": "tuntutan",
            "reshuffle": "perombakan", "rotasi": "pergantian",
            "moratorium": "penghentian", "embargo": "larangan"
        }
        auto_synonyms.update(dataset_patterns)
        
        abbreviation_patterns = {
            "yg": "yang", "dg": "dengan", "krn": "karena", "utk": "untuk",
            "tdk": "tidak", "hrs": "harus", "sdh": "sudah", "blm": "belum",
            "dr": "dari", "ke": "ke", "pd": "pada", "ttg": "tentang",
            "spy": "supaya", "krg": "kurang", "lbh": "lebih"
        }
        auto_synonyms.update(abbreviation_patterns)
        
        return auto_synonyms
        
    except Exception as e:
        st.warning(f"Error generating auto synonyms: {e}")
        return build_enhanced_synonym_dict()

# Dictionary sinonim yang akan digunakan
SYNONYM_DICT = {}

def normalize_synonyms(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    words = text.split()
    normalized_words = []
    i = 0
    while i < len(words):
        word = words[i]
        if i < len(words) - 1:
            two_word = f"{word} {words[i+1]}"
            if two_word in SYNONYM_DICT:
                normalized_words.append(SYNONYM_DICT[two_word])
                i += 2
                continue
        if word in SYNONYM_DICT:
            normalized_words.append(SYNONYM_DICT[word])
        else:
            normalized_words.append(word)
        i += 1
    return ' '.join(normalized_words)

def is_meaning_paraphrase(text1, text2, similarity_threshold=0.78):
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    emb1 = bert_model.encode(norm1, convert_to_tensor=True)
    emb2 = bert_model.encode(norm2, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb1, emb2).item()
    matcher = SequenceMatcher(None, norm1.split(), norm2.split())
    structural_sim = matcher.ratio()
    combined_score = 0.8 * semantic_sim + 0.2 * structural_sim
    return combined_score >= similarity_threshold

def enhanced_entity_matching(entities1, entities2, threshold=75):
    if not entities1 or not entities2:
        return False
    for ent1 in entities1:
        for ent2 in entities2:
            if fuzz.token_set_ratio(ent1, ent2) >= threshold:
                return True
    return False

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stop_words.discard('akan')
    stemmer = StemmerFactory().create_stemmer()
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = normalize_synonyms(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens if len(w) > 3]
    return ' '.join(tokens)

def extract_entities(text):
    doc = nlp(text)
    entities = set(ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE'])
    text_lower = text.lower()
    for party in POLITICAL_PARTIES:
        if party in text_lower.split():
            entities.add(party)
    return entities

def extract_political_parties(text):
    found_parties = set()
    text_lower = text.lower()
    temp_words = text_lower.split()
    normalized_temp_words = []
    for word in temp_words:
        normalized_temp_words.append(SYNONYM_DICT.get(word, word)) 
    processed_text = " ".join(normalized_temp_words)
    for party_name_or_alias in POLITICAL_PARTIES:
        if party_name_or_alias in processed_text:
            found_parties.add(party_name_or_alias)
    canonical_parties = set()
    for found_party in found_parties:
        if found_party in SYNONYM_DICT and SYNONYM_DICT[found_party] in POLITICAL_PARTIES:
            canonical_parties.add(SYNONYM_DICT[found_party])
        elif found_party in POLITICAL_PARTIES:
            canonical_parties.add(found_party)
    return canonical_parties

# --- NEW FUNCTION FOR SIMPLE SUMMARIZATION ---
def summarize_text_simple(text, num_sentences=2):
    """
    Summarizes a text by taking the first 'num_sentences' sentences.
    Assumes sentences are separated by periods.
    """
    if not text or pd.isna(text):
        return "Tidak ada rangkuman."
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty strings that might result from split
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "Tidak ada rangkuman."

    # Join the first 'num_sentences' sentences
    summary = " ".join(sentences[:num_sentences])
    
    # Ensure it ends with a period if it doesn't already
    if summary and summary[-1] not in ['.', '!', '?']:
        summary += "."
        
    return summary


# Load dan proses dataset
@st.cache_data
def load_data():
    global SYNONYM_DICT
    with st.spinner('Memuat dataset...'):
        df = pd.read_excel("dataset_cnn_summarized - Copy.xlsx")
        
        SYNONYM_DICT = auto_generate_synonyms(df)
        st.sidebar.markdown(f"**Loaded {len(SYNONYM_DICT)} synonym mappings**")
        
        df['text_norm'] = df['title'].apply(normalize_text)
        df['entities'] = df['title'].apply(lambda x: extract_entities(str(x)))
        df['political_parties'] = df['title'].apply(extract_political_parties)
        bert_embeddings = bert_model.encode(df['text_norm'].tolist(), convert_to_tensor=True)
        return df, bert_embeddings

# Assign loaded data and embeddings to global variables
df, bert_embeddings = load_data()

def process_batch_news(news_list, threshold_valid=0.7):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, judul in enumerate(news_list):
        if judul.strip():
            progress = (i + 1) / len(news_list)
            progress_bar.progress(progress)
            status_text.text(f'Processing {i+1}/{len(news_list)}: {judul[:50]}...')
            
            judul_norm = normalize_text(judul)
            input_entities = extract_entities(judul)
            input_parties = extract_political_parties(judul)
            input_vec = bert_model.encode(judul_norm, convert_to_tensor=True)
            
            similarity = util.pytorch_cos_sim(input_vec, bert_embeddings)[0].cpu().numpy()
            idx_max = np.argmax(similarity)
            max_score = similarity[idx_max]
            berita_mirip = df.iloc[idx_max]
            
            entities_mirip = berita_mirip['entities']
            parties_mirip = berita_mirip.get('political_parties', set())
            
            party_mismatch = input_parties != parties_mirip and (input_parties or parties_mirip)
            suspicious, prop_diff = is_suspicious_change(judul, berita_mirip['title'])
            
            if (max_score >= threshold_valid and 
                input_entities == entities_mirip and 
                not party_mismatch):
                if suspicious:
                    result = "HOAX"
                    confidence = f"Suspicious paraphrase ({prop_diff*100:.1f}% diff)"
                else:
                    result = "VALID"
                    confidence = "HIGH" if max_score > 0.85 else "MEDIUM"
            else:
                result = "HOAX"
                reasons = []
                if max_score < threshold_valid:
                    reasons.append("Low similarity")
                if input_entities != entities_mirip or party_mismatch:
                    reasons.append("Entity mismatch")
                confidence = f"Reason: {', '.join(reasons)}"
            
            # Apply simple summarization to the 'summarized' column content
            short_summary = summarize_text_simple(berita_mirip.get('summarized', 'Tidak ada rangkuman'))

            results.append({
                'Input': judul,
                'Result': result,
                'Similarity': max_score,
                'Confidence/Reason': confidence,
                'Matched_Title': berita_mirip['title'],
                'Matched_URL': berita_mirip.get('url', 'N/A'),
                'Matched_Summary_ReRangkum': short_summary # Use the re-summarized text
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def is_suspicious_change(input_text, matched_text):
    input_processed_for_diff = normalize_text(input_text)
    matched_processed_for_diff = normalize_text(matched_text)
    input_words = input_processed_for_diff.split()
    matched_words = matched_processed_for_diff.split()
    matcher = SequenceMatcher(None, input_words, matched_words)
    ratio = matcher.ratio()
    num_diff = sum(1 for tag in matcher.get_opcodes() if tag != 'equal')
    total = max(len(input_words), len(matched_words))
    prop_diff = num_diff / total if total > 0 else 0
    if ratio >= 0.98:
        return False, prop_diff
    if prop_diff < 0.1:
        return False, prop_diff
    if ratio < 0.80 and prop_diff > 0.25:
        return True, prop_diff
    return False, prop_diff

def plot_similarity_chart(similarities, df, top_n=5):
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_scores = similarities[top_indices]
    top_titles = [df.iloc[i]['title'][:50] + '...' for i in top_indices]
    fig = px.bar(
        x=top_scores, 
        y=top_titles,
        orientation='h',
        labels={'x': 'Similarity Score', 'y': 'News Title'},
        title='Top Similar News Articles',
        color=top_scores,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'range': [0, 1]},
        yaxis_title='',
        xaxis_title='Similarity Score',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def plot_similarity_distribution(similarities, max_score):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=similarities,
        nbinsx=20,
        marker_color='#90CAF9',
        opacity=0.7
    ))
    fig.add_shape(
        type="line",
        x0=max_score,
        y0=0,
        x1=max_score,
        y1=30,
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=max_score,
        y=25,
        text=f"Current: {max_score:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30
    )
    fig.update_layout(
        title='Distribution of Similarity Scores',
        xaxis_title='Similarity Score',
        yaxis_title='Number of Articles',
        height=300,
        xaxis={'range': [0, 1]},
        bargap=0.05,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def create_gauge_chart(similarity_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=similarity_score,
        title={'text': "Similarity", 'font': {'size': 24}},
        delta={'reference': 0.7, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': '#FFCDD2'},
                {'range': [0.3, 0.7], 'color': '#FFECB3'},
                {'range': [0.7, 1], 'color': '#C8E6C9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def prediksi_berita(judul_berita, threshold_valid=0.7, threshold_rekom=0.35):
    judul_norm = normalize_text(judul_berita)
    input_entities = extract_entities(judul_berita)
    input_parties = extract_political_parties(judul_berita)
    input_vec = bert_model.encode(judul_norm, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(input_vec, bert_embeddings)[0].cpu().numpy()
    idx_max = np.argmax(similarity)
    max_score = similarity[idx_max]
    berita_mirip = df.iloc[idx_max]

    entities_mirip = berita_mirip['entities']
    parties_mirip = berita_mirip.get('political_parties', set())

    party_mismatch = False
    if input_parties != parties_mirip and (input_parties or parties_mirip):
        party_mismatch = True

    suspicious, prop_diff = is_suspicious_change(judul_berita, berita_mirip['title'])

    is_para = is_meaning_paraphrase(judul_berita, berita_mirip['title'])
    entity_match = (input_entities == entities_mirip)

    if (max_score >= threshold_valid or is_para) and entity_match and not party_mismatch:
        if suspicious:
            st.markdown(f'<div class="hoax-result">', unsafe_allow_html=True)
            st.markdown(f'<div class="big-result-icon">⚠️</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">HOAX</div>', unsafe_allow_html=True)
            st.markdown(f"<p>Modifikasi struktur mencurigakan ({prop_diff*100:.1f}% kata berbeda)</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="valid-result">', unsafe_allow_html=True)
            st.markdown(f'<div class="big-result-icon">✅</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">VALID, Berita ini terverifikasi dan terpercaya</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="hoax-result">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-result-icon">❌</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">HOAX</div>', unsafe_allow_html=True)
        reasons = []
        if max_score < threshold_valid and not is_para:
            reasons.append("similarity rendah")
        elif max_score >= threshold_valid and not entity_match:
            reasons.append("entitas berbeda")
        elif max_score >= threshold_valid and party_mismatch:
            reasons.append("perbedaan partai politik")
        elif is_para and not entity_match:
            reasons.append("paraphrase namun entitas berbeda")
        elif is_para and party_mismatch:
            reasons.append("paraphrase namun perbedaan partai politik")
        
        if not reasons:
            reasons.append("tidak cocok dengan pola berita valid")

        st.markdown(f"<p>Berita tidak terverifikasi ({' dan '.join(reasons)})</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.plotly_chart(
        plot_similarity_chart(similarity, df),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if (max_score >= threshold_valid and
        input_entities == entities_mirip and
        not party_mismatch):
        # Apply simple summarization to the 'summarized' column content
        short_summary_matched = summarize_text_simple(berita_mirip.get('summarized', 'Tidak ada rangkuman'))

        if suspicious:
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            st.markdown("<h3>🔍 Informasi Detail</h3>", unsafe_allow_html=True)
            st.markdown("<p>Berita ini sangat mirip dengan berita asli namun memiliki perubahan kecil yang mencurigakan:</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Judul Asli (dari dataset)</b>: {berita_mirip['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Rangkuman Isi Berita Serupa</b>: {short_summary_matched}</p>", unsafe_allow_html=True) # Use the re-summarized text
            st.markdown(f"<p>🌐 <b>Link</b>: {berita_mirip.get('url', 'Tidak ada URL')}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            st.markdown("<h3>🔍 Informasi Detail</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Judul Berita Serupa</b>: {berita_mirip['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Rangkuman Isi Berita Serupa</b>: {short_summary_matched}</p>", unsafe_allow_html=True) # Use the re-summarized text
            st.markdown(f"<p>🌐 <b>Link</b>: {berita_mirip.get('url', 'Tidak ada URL')}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        df_valid = df[df['label'] == 0].copy()
        if not df_valid.empty:
            valid_embeddings = bert_model.encode(df_valid['text_norm'].tolist(), convert_to_tensor=True)
            sim_valid = util.pytorch_cos_sim(input_vec, valid_embeddings)[0].cpu().numpy()
            max_sim_rekom = np.max(sim_valid)

            if max_sim_rekom >= threshold_rekom:
                idx_valid = np.argmax(sim_valid)
                rekom = df_valid.iloc[idx_valid]
                
                # Apply simple summarization to the 'summarized' column content for recommendation
                short_summary_rekom = summarize_text_simple(rekom.get('summarized', 'Tidak ada rangkuman'))

                st.markdown('<div class="detail-card">', unsafe_allow_html=True)
                st.markdown("<h3>🔁 Rekomendasi Berita Valid:</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Judul</b>: {rekom['title']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Rangkuman Isi Berita Rekomendasi</b>: {short_summary_rekom}</p>", unsafe_allow_html=True) # Use the re-summarized text
                st.markdown(f"<p><b>URL</b>: {rekom.get('url', 'Tidak ada URL')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Similarity</b>: {max_sim_rekom:.4f}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-result">', unsafe_allow_html=True)
                st.markdown("⚠️ Tidak ada rekomendasi valid yang cukup mirip.", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-result">', unsafe_allow_html=True)
            st.markdown("⚠️ Tidak ada berita valid dalam dataset.", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
# Sidebar content
st.sidebar.image("https://img.icons8.com/color/96/000000/news.png", width=100)
st.sidebar.markdown("<h2>Tentang Aplikasi</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
Aplikasi ini membantu kamu mengecek apakah sebuah judul berita itu hoaks atau bukan.
Cukup masukkan judul beritanya, dan sistem kami akan membandingkan dengan berita terpercaya yang sudah diverifikasi..
""")
st.sidebar.markdown("<h2>💡 Kenapa Perlu Cek Berita?</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
Banyak berita palsu beredar yang bisa menyesatkan atau menimbulkan kepanikan. Yuk, bantu lawan hoaks mulai dari cek judulnya di sini!
""")
st.sidebar.markdown("<h2>📚 Sumber Data</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
Kami menggunakan kumpulan berita dari media terpercaya dan kecerdasan buatan untuk memberikan hasil terbaik.
""")

# Main content
st.markdown('<h1 class="main-header">🕵️‍♂️ TrueLens</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">Smarter News. Safer Minds</p>', unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📝 Input Manual", "📄 Upload File", "⚡ Batch Processing"])

with tab1:
    st.markdown("### Input Judul Berita Tunggal")
    judul_input = st.text_area("Masukkan Judul Berita:", height=100, key="single_input")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        detect_button = st.button("🔍 DETEKSI", use_container_width=True, key="single_detect")

    if detect_button:
        if judul_input.strip() == "":
            st.markdown('<div class="warning-result">', unsafe_allow_html=True)
            st.markdown("⚠️ Masukkan judul berita terlebih dahulu.", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 class="sub-header">📊 TrueLens Analysist</h2>', unsafe_allow_html=True)
            with st.spinner('Menganalisis berita...'):
                prediksi_berita(judul_input)

with tab2:
    st.markdown("### Upload File CSV/Excel")
    st.markdown("**Format file:** File harus memiliki kolom 'title' dan opsional 'summarized' yang berisi judul berita dan ringkasan")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",  
        type=['csv', 'xlsx', 'xls'],
        help="File harus memiliki kolom 'title' dan opsional 'summarized' yang berisi judul berita"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)
            
            st.success(f"File berhasil diupload! Ditemukan {len(upload_df)} baris data.")
            
            if 'title' in upload_df.columns:
                st.markdown("**Preview data:**")
                st.dataframe(upload_df.head(), use_container_width=True)
                
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    process_upload = st.button("🚀 PROSES SEMUA", use_container_width=True, key="process_upload")
                
                if process_upload:
                    st.markdown('<h2 class="sub-header">📊 Hasil Batch Processing</h2>', unsafe_allow_html=True)
                    news_titles = upload_df['title'].dropna().tolist()
                    
                    if news_titles:
                        with st.spinner('Memproses semua berita...'):
                            results_df = process_batch_news(news_titles)
                        
                        st.success(f"Selesai memproses {len(results_df)} berita!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            valid_count = len(results_df[results_df['Result'] == 'VALID'])
                            st.metric("Valid", valid_count, f"{valid_count/len(results_df)*100:.1f}%")
                        with col2:
                            hoax_count = len(results_df[results_df['Result'] == 'HOAX'])
                            st.metric("Hoax", hoax_count, f"{hoax_count/len(results_df)*100:.1f}%")
                        with col3:
                            avg_similarity = results_df['Similarity'].mean()
                            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                        with col4:
                            high_confidence = len(results_df[results_df['Confidence/Reason'] == 'HIGH'])
                            st.metric("High Confidence", high_confidence)
                        
                        st.markdown("**Hasil Detail:**")
                        # Display the new column with re-summarized text
                        st.dataframe(results_df.rename(columns={'Matched_Summary_ReRangkum': 'Rangkuman Berita Serupa'}), use_container_width=True)
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hasil CSV",
                            data=csv,
                            file_name="hoax_detection_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Tidak ada judul berita yang valid ditemukan!")
            else:
                st.error("File harus memiliki kolom 'title' yang berisi judul berita!")
                st.markdown("**Kolom yang ditemukan:**")
                st.write(list(upload_df.columns))
                
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")

with tab3:
    st.markdown("### Batch Processing - Input Multiple")
    st.markdown("Masukkan beberapa judul berita sekaligus (satu judul per baris)")
    
    batch_input = st.text_area(
        "Masukkan judul-judul berita (pisahkan dengan enter):",
        height=200,
        placeholder="Presiden akan berkunjung ke Jakarta\nMenteri kesehatan bakal resmikan rumah sakit\nDLL...",
        key="batch_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        batch_detect = st.button("🚀 PROSES BATCH", use_container_width=True, key="batch_detect")
    
    if batch_detect:
        if batch_input.strip() == "":
            st.markdown('<div class="warning-result">', unsafe_allow_html=True)
            st.markdown("⚠️ Masukkan judul berita terlebih dahulu.", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            news_list = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if news_list:
                st.markdown('<h2 class="sub-header">📊 Hasil Batch Processing</h2>', unsafe_allow_html=True)
                st.info(f"Memproses {len(news_list)} judul berita...")
                
                with st.spinner('Memproses semua berita...'):
                    results_df = process_batch_news(news_list)
                
                st.success(f"Selesai memproses {len(results_df)} berita!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    valid_count = len(results_df[results_df['Result'] == 'VALID'])
                    st.metric("Valid", valid_count, f"{valid_count/len(results_df)*100:.1f}%")
                with col2:
                    hoax_count = len(results_df[results_df['Result'] == 'HOAX'])
                    st.metric("Hoax", hoax_count, f"{hoax_count/len(results_df)*100:.1f}%")
                with col3:
                    avg_similarity = results_df['Similarity'].mean()
                    st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                with col4:
                    high_confidence = len(results_df[results_df['Confidence/Reason'] == 'HIGH'])
                    st.metric("High Confidence", high_confidence)
                
                st.markdown("**Hasil Detail:**")
                # Display the new column with re-summarized text
                st.dataframe(results_df.rename(columns={'Matched_Summary_ReRangkum': 'Rangkuman Berita Serupa'}), use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil CSV",
                    data=csv,
                    file_name="batch_hoax_detection_results.csv",
                    mime="text/csv"
                )
            else:
                st.error("Tidak ada judul berita yang valid ditemukan!")