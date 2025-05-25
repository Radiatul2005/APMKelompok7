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
from fuzzywuzzy import fuzz # Add this line


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
    
    # Base manual synonyms with more comprehensive coverage
    # Ensure all keys are consistently lowercase from the start
    base_synonyms = {
        # Modal verbs dan auxiliary verbs - ENHANCED
        "bakal": "akan", "bakalan": "akan", "hendak": "akan", "mau": "akan", 
        "akan": "akan", "mesti": "harus", "kudu": "harus", "wajib": "harus",
        
        # Common verbs - ENHANCED
        "bilang": "kata", "ucap": "kata", "sebut": "kata", "tutur": "kata",
        "omong": "kata", "cerita": "kata", "ungkap": "kata", "sampaikan": "kata",
        
        # Negations - ENHANCED
        "gak": "tidak", "ga": "tidak", "nggak": "tidak", "enggak": "tidak", 
        "tak": "tidak", "ndak": "tidak", "nda": "tidak", "bukan": "tidak",
        
        # Time indicators - ENHANCED
        "lagi": "sedang", "udah": "sudah", "udahan": "sudah", "dah": "sudah", 
        "telah": "sudah", "pernah": "sudah", "sempat": "sudah",
        "esok": "besok", "kemarin": "kemarin", "tadi": "kemarin", "tempo": "lalu",
        
        # Conjunctions - ENHANCED
        "tapi": "tetapi", "namun": "tetapi", "akan tetapi": "tetapi", "cuma": "tetapi",
        "kecuali": "tetapi", "selain": "kecuali", "dan": "dan", "serta": "dan",
        
        # Formal vs informal - ENHANCED
        "gimana": "bagaimana", "kenapa": "mengapa", "dimana": "di mana",
        "kayak": "seperti", "kaya": "seperti", "kayaknya": "sepertinya",
        "macam": "seperti", "mirip": "seperti", "ibarat": "seperti",
        
        # Intensifiers - ENHANCED
        "banget": "sangat", "sekali": "sangat", "amat": "sangat", "bener": "sangat",
        "parah": "sangat", "ekstrem": "sangat", "luar biasa": "sangat",
        
        # Prepositions - ENHANCED
        "sama": "dengan", "ama": "dengan", "ma": "dengan", "bareng": "dengan",
        "bersama": "dengan", "beserta": "dengan",
        
        # Government terms - ENHANCED
        "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", 
        "pemkab": "pemerintah kabupaten", "presiden": "presiden",
        "menteri": "menteri", "gubernur": "gubernur", "bupati": "bupati",
        "walikota": "walikota", "camat": "camat", "lurah": "lurah",
        
        # Numbers - ENHANCED
        "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
        "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10",
        "puluhan": "banyak", "ratusan": "banyak", "ribuan": "banyak",
        
        # Media terms - ENHANCED
        "kabar": "berita", "info": "informasi", "laporan": "berita", "warta": "berita",
        "news": "berita", "breaking": "terbaru", "update": "terbaru", "terkini": "terbaru",
        
        # Action verbs - ENHANCED
        "datang": "tiba", "pergi": "berangkat", "pulang": "kembali", "balik": "kembali",
        "cabut": "pergi", "nongol": "datang", "muncul": "datang", "hadir": "datang",
        
        # Adjectives - ENHANCED
        "bagus": "baik", "jelek": "buruk", "gede": "besar", "kecil": "kecil",
        "mantap": "baik", "keren": "baik", "buruk": "buruk", "parah": "buruk",
        "oke": "baik", "ok": "baik", "fine": "baik",
        
        # Slang and colloquial - ENHANCED
        "doi": "dia", "nyokap": "ibu", "bokap": "ayah", "ortu": "orang tua",
        "gue": "saya", "gw": "saya", "ane": "saya", "lu": "kamu", "lo": "kamu",
        "ente": "kamu", "bro": "saudara", "sis": "saudara",
        
        # Common Indonesian expressions
        "ngaku": "mengaku", "ngomong": "bicara", "ngasih": "memberi",
        "ngambil": "mengambil", "ngeliat": "melihat", "ngedenger": "mendengar",
        
        # Regional variations
        "nggih": "ya", "injih": "ya", "iya": "ya", "yoi": "ya", "yup": "ya",
        "enggeh": "ya", "oke": "ya", "siap": "ya",
        
        # Technology and modern terms
        "gadget": "perangkat", "smartphone": "ponsel", "laptop": "komputer",
        "online": "daring", "offline": "luring", "update": "pembaruan",
        
        # Economic terms
        "duit": "uang", "perak": "uang", "cuan": "keuntungan", "untung": "keuntungan",
        "rugi": "kerugian", "bangkrut": "pailit", "sukses": "berhasil"
    }
    
    return base_synonyms

# Auto-generate additional synonyms from dataset patterns - ENHANCED
@st.cache_data  
def auto_generate_synonyms(_df):
    """Automatically detect synonym patterns from dataset with enhanced coverage"""
    try:
        # Get base synonyms
        auto_synonyms = build_enhanced_synonym_dict().copy()
        
        # Analyze word frequency patterns from dataset
        all_words = []
        word_pairs = []
        
        for title in _df['title'].dropna():
            words = str(title).lower().split()
            clean_words = [re.sub(r'[^\w]', '', w) for w in words if len(w) > 2]
            all_words.extend(clean_words)
            
            # Look for potential synonym pairs in same sentence (this logic is still in place)
            for i, word in enumerate(clean_words):
                for j, other_word in enumerate(clean_words):
                    if i != j and abs(i-j) <= 3:  # Words close to each other
                        word_pairs.append((word, other_word))
        
        word_freq = pd.Series(all_words).value_counts()
        
        # Add context-specific patterns from dataset analysis
        dataset_patterns = {
            # Patterns likely found in Indonesian news
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
        
        # Add common abbreviation patterns
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

# Enhanced synonym normalization function
def normalize_synonyms(text):
    """Enhanced normalize synonyms in Indonesian text to reduce paraphrase variations"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    words = text.split()
    normalized_words = []
    
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check for multi-word synonyms first (e.g., "akan tetapi" -> "tetapi")
        if i < len(words) - 1:
            two_word = f"{word} {words[i+1]}"
            if two_word in SYNONYM_DICT:
                normalized_words.append(SYNONYM_DICT[two_word])
                i += 2  # Skip next word as it's part of the phrase
                continue
        
        # Check for single word synonyms
        if word in SYNONYM_DICT:
            normalized_words.append(SYNONYM_DICT[word])
        else:
            # Handle common patterns that might not be in dictionary
            # Remove common prefixes/suffixes that don't change meaning significantly
            # NOTE: These explicit prefix/suffix removals might be redundant if stemming is strong
            # clean_word = word
            # if word.startswith('me') and len(word) > 3:
            #     base_word = word[2:]  # Remove 'me' prefix
            #     if base_word in SYNONYM_DICT:
            #         clean_word = SYNONYM_DICT[base_word]
            # elif word.startswith('di') and len(word) > 3:
            #     base_word = word[2:]  # Remove 'di' prefix  
            #     if base_word in SYNONYM_DICT:
            #         clean_word = SYNONYM_DICT[base_word]
            
            # For now, just append the original word if no single or multi-word synonym found
            normalized_words.append(word) # Changed from clean_word to word, as prefix stripping should be handled by stemmer
        
        i += 1
    
    return ' '.join(normalized_words)

def is_meaning_paraphrase(text1, text2, similarity_threshold=0.78): # Changed from 0.82 to 0.78
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    emb1 = bert_model.encode(norm1, convert_to_tensor=True)
    emb2 = bert_model.encode(norm2, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb1, emb2).item()

    matcher = SequenceMatcher(None, norm1.split(), norm2.split())
    structural_sim = matcher.ratio()

    # The weighting of structural_sim might be too high for rephrasing
    combined_score = 0.8 * semantic_sim + 0.2 * structural_sim # Adjusted weights: more semantic, less structural

    return combined_score >= similarity_threshold

def enhanced_entity_matching(entities1, entities2, threshold=75):
    """Fuzzy matching untuk entitas dengan threshold tertentu"""
    if not entities1 or not entities2:
        return False
    
    for ent1 in entities1:
        for ent2 in entities2:
            # Gunakan fuzzy matching untuk entitas
            if fuzz.token_set_ratio(ent1, ent2) >= threshold:
                return True
    return False

# Enhanced text normalization with better synonym handling
def normalize_text(text):
    """Enhanced text normalization with improved synonym handling"""
    if not text or pd.isna(text):
        return ""
    
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    # Jangan hapus kata penting seperti 'akan'
    stop_words.discard('akan')
    stemmer = StemmerFactory().create_stemmer()
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Normalisasi sinonim dulu
    text = normalize_synonyms(text)
    
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    
    # Stemming lebih hati-hati
    tokens = [stemmer.stem(w) for w in tokens if len(w) > 3]  # Hanya stem kata panjang
    
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
    found_parties = set()
    text_lower = text.lower() # Start with raw lower text

    # Apply synonym normalization only to individual words or small phrases to detect party aliases
    # This doesn't apply full normalization (stemming, stop words) which would disrupt multi-word names
    temp_words = text_lower.split()
    normalized_temp_words = []
    for word in temp_words:
        # Check for multi-word synonyms first if they exist in SYNONYM_DICT (e.g. "akan tetapi")
        # For party names, this is less likely unless you have "partai a" -> "partai b"
        # The main goal here is single-word abbreviation like "pkb" -> "partai kebangkitan bangsa"
        normalized_temp_words.append(SYNONYM_DICT.get(word, word)) # Get mapped value or original word

    processed_text = " ".join(normalized_temp_words) # Reconstruct for phrase matching

    # Now, iterate through the canonical POLITICAL_PARTIES list.
    # Check if the full name (or common abbreviation) is present in the processed text.
    for party_name_or_alias in POLITICAL_PARTIES:
        if party_name_or_alias in processed_text:
            # Add the canonical full name to the set of found parties for consistency
            # This requires a reverse lookup or a mapping from alias to canonical name if needed,
            # or just add whatever form is in POLITICAL_PARTIES.
            # For simplicity, let's just add the matched form from POLITICAL_PARTIES to found_parties
            found_parties.add(party_name_or_alias)

    # To ensure consistent output like "partai kebangkitan bangsa"
    # Create a mapping from any alias to its canonical full name if you want a single representation:
    canonical_parties = set()
    for found_party in found_parties:
        # If the found party is an alias (e.g., "pkb"), look up its canonical form.
        # This assumes your SYNONYM_DICT maps "pkb" -> "partai kebangkitan bangsa"
        # and "partai kebangkitan bangsa" is in POLITICAL_PARTIES.
        if found_party in SYNONYM_DICT and SYNONYM_DICT[found_party] in POLITICAL_PARTIES:
            canonical_parties.add(SYNONYM_DICT[found_party])
        elif found_party in POLITICAL_PARTIES: # It was already a canonical name or a direct match
            canonical_parties.add(found_party)

    return canonical_parties


# Load dan proses dataset
@st.cache_data
def load_data():
    global SYNONYM_DICT
    with st.spinner('Memuat dataset...'):
        df = pd.read_excel("dataset_cnn_summarized - Copy.xlsx")
        
        # Generate enhanced synonym dict from dataset
        SYNONYM_DICT = auto_generate_synonyms(df)
        
        # Show synonym dict info in sidebar for debugging
        st.sidebar.markdown(f"**Loaded {len(SYNONYM_DICT)} synonym mappings**")
        
        df['text_norm'] = df['title'].apply(normalize_text)
        df['entities'] = df['title'].apply(lambda x: extract_entities(str(x)))
        df['political_parties'] = df['title'].apply(extract_political_parties)
        bert_embeddings = bert_model.encode(df['text_norm'].tolist(), convert_to_tensor=True)
        return df, bert_embeddings

# Enhanced batch processing function with better paraphrase handling
def process_batch_news(news_list, threshold_valid=0.7):
    """Process multiple news titles at once with enhanced paraphrase detection"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, judul in enumerate(news_list):
        if judul.strip():
            # Update progress
            progress = (i + 1) / len(news_list)
            progress_bar.progress(progress)
            status_text.text(f'Processing {i+1}/{len(news_list)}: {judul[:50]}...')
            
            # Process single news with enhanced normalization
            judul_norm = normalize_text(judul)
            input_entities = extract_entities(judul)
            input_parties = extract_political_parties(judul)
            input_vec = bert_model.encode(judul_norm, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(input_vec, bert_embeddings)[0].cpu().numpy()
            idx_max = np.argmax(similarity)
            max_score = similarity[idx_max]
            berita_mirip = df.iloc[idx_max]
            
            entities_mirip = berita_mirip['entities']
            parties_mirip = berita_mirip.get('political_parties', set())
            
            # Enhanced determination with paraphrase consideration
            party_mismatch = input_parties != parties_mirip and (input_parties or parties_mirip)
            
            # Check for suspicious paraphrase patterns
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
            
            results.append({
                'Input': judul,
                'Result': result,
                'Similarity': max_score,
                'Confidence/Reason': confidence,
                'Matched_Title': berita_mirip['title'],
                'Matched_URL': berita_mirip.get('url', 'N/A')
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# ---
# CRITICAL FIX APPLIED HERE
# Enhanced suspicious change detection for paraphrases
def is_suspicious_change(input_text, matched_text):
    input_processed_for_diff = normalize_text(input_text)
    matched_processed_for_diff = normalize_text(matched_text)

    input_words = input_processed_for_diff.split()
    matched_words = matched_processed_for_diff.split()

    matcher = SequenceMatcher(None, input_words, matched_words)
    ratio = matcher.ratio()

    num_diff = sum(1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != 'equal')
    total = max(len(input_words), len(matched_words))
    prop_diff = num_diff / total if total > 0 else 0

    if ratio >= 0.98:
        return False, prop_diff

    if prop_diff < 0.1:
        return False, prop_diff

    if ratio < 0.80 and prop_diff > 0.25:
        return True, prop_diff

    return False, prop_diff

    # If the normalized text is highly similar (e.g., > 0.85) AND has very few differing blocks,
    # it's likely a valid paraphrase (synonym, slight reordering) that normalization mostly handled.
    if ratio > 0.85 and prop_diff < 0.15: # Tuned thresholds
        return False, prop_diff

    return True, prop_diff

# ---

# Create similarity bar chart for top matches
def plot_similarity_chart(similarities, df, top_n=5):
    # Get top N matches
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_scores = similarities[top_indices]
    top_titles = [df.iloc[i]['title'][:50] + '...' for i in top_indices]
    
    # Create bar chart with Plotly
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

# Create similarity distribution histogram
def plot_similarity_distribution(similarities, max_score):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=similarities,
        nbinsx=20,
        marker_color='#90CAF9',
        opacity=0.7
    ))
    
    # Add vertical line for current similarity score
    fig.add_shape(
        type="line",
        x0=max_score,
        y0=0,
        x1=max_score,
        y1=30,  # Will be updated by autorange
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

# Define function for creating the gauge chart
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
                {'range': [0, 0.3], 'color': '#FFCDD2'},  # Low similarity - red
                {'range': [0.3, 0.7], 'color': '#FFECB3'},  # Medium similarity - yellow
                {'range': [0.7, 1], 'color': '#C8E6C9'}    # High similarity - green
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
            st.markdown(f'<div class="result-title">VALID</div>', unsafe_allow_html=True)
            st.markdown("<p>Berita ini terverifikasi dan terpercaya</p>", unsafe_allow_html=True)
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

    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Detail Analisis</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-container'><div class='metric-label'>Similarity Score</div><div class='metric-value'>{max_score:.4f}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Judul Paling Mirip di Dataset:</b> {berita_mirip['title']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Judul Paling Mirip (Setelah Normalisasi):</b> {normalize_text(berita_mirip['title'])}</p>", unsafe_allow_html=True)

    st.plotly_chart(
        plot_similarity_chart(similarity, df),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if (max_score >= threshold_valid and
        input_entities == entities_mirip and
        not party_mismatch):
        if suspicious:
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            st.markdown("<h3>🔍 Informasi Detail</h3>", unsafe_allow_html=True)
            st.markdown("<p>Berita ini sangat mirip dengan berita asli namun memiliki perubahan kecil yang mencurigakan:</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Judul Asli (dari dataset)</b>: {berita_mirip['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>🌐 <b>Link</b>: {berita_mirip.get('url', 'Tidak ada URL')}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            st.markdown("<h3>🔍 Informasi Detail</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>🌐 <b>Link</b>: {berita_mirip.get('url', 'Tidak ada URL')}</p>", unsafe_allow_html=True)
            try:
                tahun = pd.to_datetime(berita_mirip.get('date', None)).year
                st.markdown(f"<p>📅 <b>Tahun Terbit</b>: {tahun}</p>", unsafe_allow_html=True)
            except:
                st.markdown("<p>⚠️ Tahun tidak bisa diambil.</p>", unsafe_allow_html=True)
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

                st.markdown('<div class="detail-card">', unsafe_allow_html=True)
                st.markdown("<h3>🔁 Rekomendasi Berita Valid:</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Judul</b>: {rekom['title']}</p>", unsafe_allow_html=True)
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
Aplikasi ini menggunakan NLP dan model BERT untuk mendeteksi berita hoax 
dengan membandingkan judul berita input dengan database berita yang terverifikasi.

**Fitur Utama:**
- Deteksi kemiripan dengan berita terverifikasi
- Analisis entitas (nama, organisasi, lokasi)
- Deteksi partai politik
- **Normalisasi sinonim otomatis dari dataset**
- **Batch processing untuk multiple berita**
- **Upload file CSV/Excel**
- Visualisasi similarity score
- Rekomendasi berita valid terkait
""")

st.sidebar.markdown("<h2>Cara Penggunaan</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Input Manual:**
1. Masukkan judul berita di tab "Input Manual"
2. Klik tombol "Deteksi"

**Upload File:**
1. Siapkan file CSV/Excel dengan kolom 'title'
2. Upload di tab "Upload File"  
3. Klik "Proses Semua"

**Batch Processing:**
1. Masukkan multiple judul di tab "Batch Processing"
2. Pisahkan setiap judul dengan enter
3. Klik "Proses Batch"
""")

st.sidebar.markdown("<h2>Info Threshold</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
- **Valid**: Similarity ≥ 0.7 dan entitas cocok
- **Hoax**: Similarity < 0.7 atau entitas berbeda
- **Rekomendasi**: Similarity ≥ 0.35
- **Normalisasi Sinonim**: Menangani kata seperti "bakal"→"akan"
""")

# Load models when app starts
bert_model, nlp = load_models()
df, bert_embeddings = load_data()

# Main content
st.markdown('<h1 class="main-header">🕵️‍♂️ DETEKSI BERITA HOAX</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">Powered by NLP, BERT, dan Analisis Entitas dengan Normalisasi Sinonim</p>', unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📝 Input Manual", "📄 Upload File", "⚡ Batch Processing"])

with tab1:
    st.markdown("### Input Judul Berita Tunggal")
    # Input section with improved styling
    judul_input = st.text_area("Masukkan Judul Berita:", height=100, key="single_input")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        detect_button = st.button("🔍 DETEKSI", use_container_width=True, key="single_detect")

    # Results section for single input
    if detect_button:
        if judul_input.strip() == "":
            st.markdown('<div class="warning-result">', unsafe_allow_html=True)
            st.markdown("⚠️ Masukkan judul berita terlebih dahulu.", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 class="sub-header">📊 Hasil Analisis</h2>', unsafe_allow_html=True)
            with st.spinner('Menganalisis berita...'):
                prediksi_berita(judul_input)

with tab2:
    st.markdown("### Upload File CSV/Excel")
    st.markdown("**Format file:** File harus memiliki kolom 'title' yang berisi judul berita")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",  
        type=['csv', 'xlsx', 'xls'],
        help="File harus memiliki kolom 'title' yang berisi judul berita"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            if uploaded_file.name.endswith('.csv'):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)
            
            st.success(f"File berhasil diupload! Ditemukan {len(upload_df)} baris data.")
            
            # Show preview
            if 'title' in upload_df.columns:
                st.markdown("**Preview data:**")
                st.dataframe(upload_df.head(), use_container_width=True)
                
                # Process button
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    process_upload = st.button("🚀 PROSES SEMUA", use_container_width=True, key="process_upload")
                
                if process_upload:
                    st.markdown('<h2 class="sub-header">📊 Hasil Batch Processing</h2>', unsafe_allow_html=True)
                    
                    # Process all titles
                    news_titles = upload_df['title'].dropna().tolist()
                    
                    if news_titles:
                        with st.spinner('Memproses semua berita...'):
                            results_df = process_batch_news(news_titles)
                        
                        # Display results
                        st.success(f"Selesai memproses {len(results_df)} berita!")
                        
                        # Summary statistics
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
                        
                        # Results table
                        st.markdown("**Hasil Detail:**")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
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
            # Split input by lines
            news_list = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if news_list:
                st.markdown('<h2 class="sub-header">📊 Hasil Batch Processing</h2>', unsafe_allow_html=True)
                st.info(f"Memproses {len(news_list)} judul berita...")
                
                with st.spinner('Memproses semua berita...'):
                    results_df = process_batch_news(news_list)
                
                # Display results (same as upload section)
                st.success(f"Selesai memproses {len(results_df)} berita!")
                
                # Summary statistics
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
                
                # Results table
                st.markdown("**Hasil Detail:**")
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil CSV",
                    data=csv,
                    file_name="batch_hoax_detection_results.csv",
                    mime="text/csv"
                )
            else:
                st.error("Tidak ada judul berita yang valid ditemukan!")

# Footer (commented out as in your original code)
# st.markdown("""
# <div style="text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;">
#     <p>© 2025 Hoax Detector System | Developed with ❤️ using Streamlit and BERT</p>
# </div>
# """, unsafe_allow_html=True)