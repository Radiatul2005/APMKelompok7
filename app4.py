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
    "pkb", "gerindra", "golkar", "pdip", "pdi-p", "demokrat", "pan", "pks",  
    "nasdem", "ppp", "hanura", "psi", "perindo", "pkpi", "berkarya"
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
        word_pairs = [] # Keep this for potential future advanced detection
        
        for title in _df['title'].dropna():
            words = str(title).lower().split()
            clean_words = [re.sub(r'[^\w]', '', w) for w in words if len(w) > 2]
            all_words.extend(clean_words)
            
            # Look for potential synonym pairs in same sentence (this logic is still in place)
            for i, word in enumerate(clean_words):
                for j, other_word in enumerate(clean_words):
                    if i != j and abs(i-j) <= 3:  # Words close to each other
                        word_pairs.append((word, other_word))
        
        # word_freq = pd.Series(all_words).value_counts() # Not used for direct synonym generation here
        
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
STOP_WORDS = set() # Global stop words

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
        found_multi_word = False
        for k in range(min(3, len(words) - i), 1, -1): # Check for 3-word, then 2-word phrases
            phrase = " ".join(words[i : i + k])
            if phrase in SYNONYM_DICT:
                normalized_words.append(SYNONYM_DICT[phrase])
                i += k
                found_multi_word = True
                break
        
        if found_multi_word:
            continue
            
        # Check for single word synonyms
        if word in SYNONYM_DICT:
            normalized_words.append(SYNONYM_DICT[word])
        else:
            normalized_words.append(word)
        
        i += 1
    
    return ' '.join(normalized_words)

# Enhanced text normalization with better synonym handling
def normalize_text(text):
    """Enhanced text normalization with improved synonym handling"""
    if not text or pd.isna(text):
        return ""
    
    stemmer = StemmerFactory().create_stemmer()
    
    text = str(text).lower()
    
    # Clean punctuation but preserve word boundaries
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)   # Multiple spaces to single space
    text = text.strip() # Remove leading/trailing spaces

    # Apply synonym normalization BEFORE stemming for better effectiveness
    # Stemming might change words so much that synonym detection becomes hard
    text = normalize_synonyms(text)
    
    # Tokenize and filter
    tokens = text.split()
    
    # Filter stop words and empty/short tokens
    tokens = [w for w in tokens if w and w not in STOP_WORDS] 
    
    # Apply stemming
    tokens = [stemmer.stem(w) for w in tokens]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token and token not in seen: # Ensure token is not empty after stemming
            seen.add(token)
            unique_tokens.append(token)
    
    return ' '.join(unique_tokens)

# Ekstrak entitas dengan penekanan pada partai politik
def extract_entities(text):
    doc = nlp(text)
    # Extract standard named entities, including DATE and CARDINAL for more comprehensive entity comparison
    entities = set(ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL']) 
    
    # Add custom detection for Indonesian political parties
    text_lower = text.lower()
    # Check for exact word matches for political parties
    for party in POLITICAL_PARTIES:
        if re.search(r'\b' + re.escape(party) + r'\b', text_lower):
            entities.add(party)
    
    return entities

# Extract political parties specifically
def extract_political_parties(text):
    text_lower = text.lower()
    found_parties = set()
    for party in POLITICAL_PARTIES:
        if re.search(r'\b' + re.escape(party) + r'\b', text_lower):
            found_parties.add(party)
    return found_parties

# Load dan proses dataset
@st.cache_data
def load_data():
    global SYNONYM_DICT
    global STOP_WORDS
    with st.spinner('Memuat dataset dan memproses...'):
        df = pd.read_excel("dataset_cnn_summarized - Copy.xlsx")
        
        # Initialize stop words
        stop_factory = StopWordRemoverFactory()
        STOP_WORDS = set(stop_factory.get_stop_words())

        # Generate enhanced synonym dict from dataset
        SYNONYM_DICT = auto_generate_synonyms(df)
        
        # Show synonym dict info in sidebar for debugging
        st.sidebar.markdown(f"**Loaded {len(SYNONYM_DICT)} synonym mappings**")
        
        df['text_norm'] = df['title'].apply(normalize_text)
        df['entities'] = df['title'].apply(lambda x: extract_entities(str(x)))
        df['political_parties'] = df['title'].apply(extract_political_parties)
        
        # Filter out rows where 'text_norm' might be empty or problematic before encoding
        # This prevents errors if normalization results in empty strings
        df_filtered = df[df['text_norm'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()
        
        # Handle cases where all entries might be filtered out
        if df_filtered.empty:
            st.error("Dataset kosong setelah normalisasi! Pastikan kolom 'title' berisi teks yang valid.")
            return pd.DataFrame(), None # Return empty DataFrame and None for embeddings
        
        bert_embeddings = bert_model.encode(df_filtered['text_norm'].tolist(), convert_to_tensor=True)
        return df_filtered, bert_embeddings

# --- NEW FUNCTION FOR PARAPHRASE DETECTION ---
def is_paraphrase_of(input_text, matched_text, bert_model, nlp_spacy, 
                     threshold_semantic=0.60, # Adjusted for more flexibility
                     threshold_structural=0.50, # Adjusted for more flexibility
                     threshold_entity_match=0.75): # Adjusted for more flexibility
    """
    Menentukan apakah input_text adalah parafrase dari matched_text.
    Menggabungkan kemiripan semantik, struktural, dan entitas.
    """
    
    # 1. Normalisasi Teks (termasuk sinonim, stemming, stop words)
    input_norm = normalize_text(input_text)
    matched_norm = normalize_text(matched_text)
    
    # If normalized texts are identical, it's a strong paraphrase
    if input_norm == matched_norm and input_norm != "": 
        return True, "Identical after normalization"

    # 2. Kemiripan Semantik (BERT)
    # Handle empty normalized text to avoid BERT errors
    if not input_norm or not matched_norm:
        return False, "Empty normalized text after processing"

    input_vec = bert_model.encode(input_norm, convert_to_tensor=True)
    matched_vec = bert_model.encode(matched_norm, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(input_vec, matched_vec).item()
    
    if semantic_similarity < threshold_semantic:
        return False, f"Low semantic similarity ({semantic_similarity:.2f})"

    # 3. Kemiripan Struktural (After Normalization)
    input_words = input_norm.split()
    matched_words = matched_norm.split()
    
    if not input_words or not matched_words:
        structural_similarity = 0.0
    else:
        matcher = SequenceMatcher(None, input_words, matched_words)
        structural_similarity = matcher.ratio()
    
    if structural_similarity < threshold_structural:
        return False, f"Low structural similarity ({structural_similarity:.2f} after normalization)"
        
    # 4. Pencocokan Entitas
    input_entities = extract_entities(input_text)
    matched_entities = extract_entities(matched_text)
    
    if not input_entities and not matched_entities:
        entity_match_score = 1.0 
    else:
        intersection = len(input_entities.intersection(matched_entities))
        union = len(input_entities.union(matched_entities))
        entity_match_score = intersection / union if union > 0 else 0.0

    if entity_match_score < threshold_entity_match:
        return False, f"Entity mismatch ({entity_match_score:.2f}) - Input: {input_entities}, Matched: {matched_entities}"

    return True, "Strong paraphrase"
# --- END NEW FUNCTION ---

# Enhanced batch processing function with better paraphrase handling
def process_batch_news(news_list, threshold_valid=0.60): # Use adjusted threshold here as well
    """Process multiple news titles at once with enhanced paraphrase detection"""
    results = []
    
    # Ensure df and bert_embeddings are loaded
    if df is None or bert_embeddings is None:
        st.error("Dataset tidak berhasil dimuat. Mohon periksa file dataset Anda.")
        return pd.DataFrame()

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
            
            # Skip if normalized text is empty
            if not judul_norm.strip():
                results.append({
                    'Input': judul,
                    'Result': "SKIP",
                    'Similarity': 0.0,
                    'Confidence/Reason': "Normalized text is empty",
                    'Matched_Title': 'N/A',
                    'Matched_URL': 'N/A'
                })
                continue
            
            input_parties = extract_political_parties(judul) # Keep this for specific party check
            input_vec = bert_model.encode(judul_norm, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(input_vec, bert_embeddings)[0].cpu().numpy()
            idx_max = np.argmax(similarity)
            max_score = similarity[idx_max]
            berita_mirip = df.iloc[idx_max]
            
            parties_mirip = berita_mirip.get('political_parties', set())
            
            # Check for political party mismatch specifically
            party_mismatch = False
            if input_parties != parties_mirip and (input_parties or parties_mirip):
                party_mismatch = True

            # Use the new paraphrase function
            is_paraphrase, paraphrase_reason = is_paraphrase_of(
                judul, 
                berita_mirip['title'], 
                bert_model, 
                nlp,
                threshold_semantic=threshold_valid, 
                threshold_structural=0.50, # Use adjusted threshold here as well
                threshold_entity_match=0.75 # Use adjusted threshold here as well
            )
            
            if is_paraphrase and not party_mismatch:
                result = "VALID"
                confidence = f"Strong paraphrase ({paraphrase_reason})"
            else:
                result = "HOAX"
                reasons = []
                if not is_paraphrase:
                    reasons.append(f"bukan parafrase kuat ({paraphrase_reason})")
                if party_mismatch:
                    reasons.append("entitas partai politik berbeda")
                confidence = f"Reason: {', '.join(reasons)}"
            
            results.append({
                'Input': judul,
                'Result': result,
                'Similarity': max_score, # Still show the raw BERT similarity
                'Confidence/Reason': confidence,
                'Matched_Title': berita_mirip['title'],
                'Matched_URL': berita_mirip.get('url', 'N/A')
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# Create similarity bar chart for top matches
def plot_similarity_chart(similarities, df, top_n=5):
    # Ensure df is not empty
    if df.empty:
        st.warning("Tidak ada data untuk menampilkan grafik kemiripan.")
        return go.Figure()

    # Get top N matches
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Filter out invalid indices if there are fewer than top_n matches
    valid_indices = [idx for idx in top_indices if idx < len(df)]
    
    top_scores = similarities[valid_indices]
    top_titles = [df.iloc[i]['title'][:50] + '...' for i in valid_indices]
    
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

# Define function for creating the gauge chart (kept for display if needed, but not used in new logic)
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

# Enhanced prediction function with better paraphrase handling
def prediksi_berita(judul_berita, threshold_valid=0.60, threshold_rekom=0.35): # Using adjusted default threshold
    """Enhanced prediction with better paraphrase detection and handling"""
    
    # Ensure df and bert_embeddings are loaded
    if df is None or bert_embeddings is None:
        st.error("Dataset tidak berhasil dimuat. Mohon periksa file dataset Anda.")
        return

    # Enhanced text processing
    judul_norm = normalize_text(judul_berita)
    input_parties = extract_political_parties(judul_berita)
    
    # Handle case where normalized text might be empty
    if not judul_norm.strip():
        st.markdown('<div class="warning-result">', unsafe_allow_html=True)
        st.markdown("⚠️ Judul berita terlalu pendek atau tidak mengandung informasi yang cukup setelah normalisasi. Tidak bisa diproses.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    input_vec = bert_model.encode(judul_norm, convert_to_tensor=True)
    
    # Hitung similarity
    similarity = util.pytorch_cos_sim(input_vec, bert_embeddings)[0].cpu().numpy()
    idx_max = np.argmax(similarity)
    max_score = similarity[idx_max]
    berita_mirip = df.iloc[idx_max]
    
    parties_mirip = berita_mirip.get('political_parties', set())
    
    # Check for political party mismatch specifically
    party_mismatch = False
    if input_parties != parties_mirip and (input_parties or parties_mirip):
        party_mismatch = True
    
    # --- Perubahan Utama di Sini ---
    # Gunakan fungsi is_paraphrase_of yang baru
    is_paraphrase, paraphrase_reason = is_paraphrase_of(
        judul_berita, 
        berita_mirip['title'], 
        bert_model, 
        nlp,
        threshold_semantic=threshold_valid, 
        threshold_structural=0.50, # Use adjusted threshold here
        threshold_entity_match=0.75 # Use adjusted threshold here
    )
    
    # --- Debugging Output (Akan muncul di konsol terminal tempat Streamlit dijalankan) ---
    print(f"\n--- Debugging Prediksi Berita ---")
    print(f"Input Judul Asli: '{judul_berita}'")
    print(f"Input Judul Normal: '{judul_norm}'")
    print(f"Matched Judul Asli: '{berita_mirip['title']}'")
    print(f"Matched Judul Normal: '{normalize_text(berita_mirip['title'])}'")
    
    # Re-calculate and print scores with fixed thresholds used in is_paraphrase_of for clarity in debug
    temp_input_norm = normalize_text(judul_berita)
    temp_matched_norm = normalize_text(berita_mirip['title'])
    
    temp_input_vec = bert_model.encode(temp_input_norm, convert_to_tensor=True)
    temp_matched_vec = bert_model.encode(temp_matched_norm, convert_to_tensor=True)
    temp_semantic_similarity = util.pytorch_cos_sim(temp_input_vec, temp_matched_vec).item()
    
    temp_input_words = temp_input_norm.split()
    temp_matched_words = temp_matched_norm.split()
    temp_structural_similarity = SequenceMatcher(None, temp_input_words, temp_matched_words).ratio() if temp_input_words and temp_matched_words else 0.0

    temp_input_entities = extract_entities(judul_berita)
    temp_matched_entities = extract_entities(berita_mirip['title'])
    temp_intersection = len(temp_input_entities.intersection(temp_matched_entities))
    temp_union = len(temp_input_entities.union(temp_matched_entities))
    temp_entity_match_score = temp_intersection / temp_union if temp_union > 0 else 0.0

    print(f"BERT Max Score (Overall Dataset): {max_score:.4f}")
    print(f"--- Scores for is_paraphrase_of with Thresholds: ---")
    print(f"  Semantic Similarity: {temp_semantic_similarity:.4f} (Threshold: {threshold_valid})")
    print(f"  Structural Similarity: {temp_structural_similarity:.4f} (Threshold: 0.50)")
    print(f"  Entity Match Score (Jaccard): {temp_entity_match_score:.4f} (Threshold: 0.75)")
    print(f"  Input Entities: {temp_input_entities}")
    print(f"  Matched Entities: {temp_matched_entities}")
    print(f"Is Paraphrase Check Result: {is_paraphrase} (Reason: {paraphrase_reason})")
    print(f"Input Parties: {input_parties}, Matched Parties: {parties_mirip}, Mismatch: {party_mismatch}")
    print(f"--- End Debug ---")
    # --- End Debugging Output ---

    # Output decision based on new paraphrase logic
    if is_paraphrase and not party_mismatch: # Ensure political parties are consistent
        st.markdown(f'<div class="valid-result">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-result-icon">✅</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">VALID</div>', unsafe_allow_html=True)
        st.markdown(f"<p>Berita ini terverifikasi dan terpercaya (Deteksi parafrase: {paraphrase_reason})</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # If not a strong paraphrase, or if there's a party mismatch, classify as HOAX.
        reasons = []
        if not is_paraphrase:
            reasons.append(f"bukan parafrase kuat ({paraphrase_reason})")
        if party_mismatch:
            reasons.append("entitas partai politik berbeda")
            
        reason_str = " dan ".join(reasons)
        
        st.markdown(f'<div class="hoax-result">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-result-icon">❌</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">HOAX</div>', unsafe_allow_html=True)
        st.markdown(f"<p>Berita tidak terverifikasi ({reason_str})</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # THEN DISPLAY THE SIMILARITY INFORMATION
    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Detail Analisis</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-container'><div class='metric-label'>Similarity Score (BERT)</div><div class='metric-value'>{max_score:.4f}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Judul Paling Mirip di Dataset:</b> {berita_mirip['title']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Judul Paling Mirip (Setelah Normalisasi):</b> {normalize_text(berita_mirip['title'])}</p>", unsafe_allow_html=True)
    
    # Display the similarity chart
    st.plotly_chart(
        plot_similarity_chart(similarity, df), 
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # NOW DISPLAY ADDITIONAL INFORMATION BASED ON THE RESULT
    if is_paraphrase and not party_mismatch: # Use the new result for display logic
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
        # Rekomendasi berita valid
        df_valid = df[df['label'] == 0].copy()
        if not df_valid.empty:
            # Ensure text_norm column exists and is not empty for valid articles
            df_valid_processed = df_valid[df_valid['text_norm'].apply(lambda x: bool(x.strip()))]
            if not df_valid_processed.empty:
                valid_embeddings = bert_model.encode(df_valid_processed['text_norm'].tolist(), convert_to_tensor=True)
                sim_valid = util.pytorch_cos_sim(input_vec, valid_embeddings)[0].cpu().numpy()
                max_sim_rekom = np.max(sim_valid)

                if max_sim_rekom >= threshold_rekom:
                    idx_valid = np.argmax(sim_valid)
                    rekom = df_valid_processed.iloc[idx_valid]
                    
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
                st.markdown("⚠️ Tidak ada berita valid yang dapat diproses dalam dataset.", unsafe_allow_html=True)
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
- **Deteksi parafrase canggih untuk akurasi lebih tinggi**
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
- **Valid (Parafrase):** Semantic Similarity ≥ 0.60, Structural Similarity ≥ 0.50, Entity Match ≥ 0.75. Entitas partai politik harus cocok.
- **Hoax:** Tidak memenuhi kriteria valid.
- **Rekomendasi:** Semantic Similarity ≥ 0.35
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
                            st.metric("Avg BERT Similarity", f"{avg_similarity:.3f}")
                        with col4:
                            # Assuming "Strong paraphrase" implies high confidence in being valid
                            strong_paraphrase_count = len(results_df[results_df['Confidence/Reason'].str.contains("Strong paraphrase")])
                            st.metric("Deteksi Parafrase Kuat", strong_paraphrase_count)
                        
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
                    st.metric("Avg BERT Similarity", f"{avg_similarity:.3f}")
                with col4:
                    strong_paraphrase_count = len(results_df[results_df['Confidence/Reason'].str.contains("Strong paraphrase")])
                    st.metric("Deteksi Parafrase Kuat", strong_paraphrase_count)
                
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