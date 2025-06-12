import pandas as pd
# import numpy as np
import re
import spacy
from difflib import SequenceMatcher
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib # To save/load the trained Naive Bayes model and vectorizer

# Custom political parties list for Indonesian context (unchanged)
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

# Global models (initialized at module level)
BERT_MODEL = None
NLP = None
NB_MODEL_PIPELINE = None # Initialize global variable here

def load_models_base():
    bert_model = SentenceTransformer("indobenchmark/indobert-base-p1")
    nlp = spacy.load("xx_ent_wiki_sm") # Consider a larger model if xx_ent_wiki_sm is insufficient for specific NER, e.g., 'id_core_news_lg' if available and performing better for Indonesian
    return bert_model, nlp

# Enhanced synonym dictionary (unchanged)
def build_enhanced_synonym_dict_base():
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

# Dictionary sinonim yang akan digunakan, initialized when data is loaded (unchanged)
SYNONYM_DICT = {}

def normalize_synonyms(text): # (unchanged)
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


def set_global_models(bert_model, nlp_model): # (unchanged)
    global BERT_MODEL, NLP
    BERT_MODEL = bert_model
    NLP = nlp_model

# NEW: Set global Naive Bayes pipeline
def set_global_nb_pipeline(pipeline):
    global NB_MODEL_PIPELINE
    NB_MODEL_PIPELINE = pipeline

def is_meaning_paraphrase(text1, text2, similarity_threshold=0.78): # (unchanged)
    if BERT_MODEL is None:
        raise RuntimeError("BERT_MODEL not initialized. Call set_global_models() first.")
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    emb1 = BERT_MODEL.encode(norm1, convert_to_tensor=True)
    emb2 = BERT_MODEL.encode(norm2, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb1, emb2).item()
    matcher = SequenceMatcher(None, norm1.split(), norm2.split())
    structural_sim = matcher.ratio()
    combined_score = 0.8 * semantic_sim + 0.2 * structural_sim
    return combined_score >= similarity_threshold

# NEW OR MODIFIED FUNCTION: compare_entities
def compare_entities(input_entities, matched_entities, required_match_ratio=0.8, fuzzy_threshold=80):
    """
    Compares two sets of entities (each entity is a string).
    Requires a high degree of overlap or fuzzy matching.
    Returns True if entities are considered a good match, False otherwise.
    
    This function prioritizes ensuring that entities are consistent.
    """
    # If both lists are empty, they match
    if not input_entities and not matched_entities:
        return True
    
    # If one is empty and the other is not, they don't match
    if not input_entities or not matched_entities:
        return False
    
    # Track how many input entities find a match in the matched entities
    matched_input_entities_count = 0
    
    # To avoid double-matching entities, use a temporary mutable list for matched_entities
    temp_matched_entities = list(matched_entities)

    for i_ent in input_entities:
        # found_match = False
        # Try to find a fuzzy match for i_ent in temp_matched_entities
        for m_ent in temp_matched_entities:
            if fuzz.token_set_ratio(i_ent, m_ent) >= fuzzy_threshold:
                matched_input_entities_count += 1
                temp_matched_entities.remove(m_ent) # Remove it so it's not matched again
                # found_match = True
                break
        
    # Special case: if there's only one entity in both, and they don't fuzzy match, it's a clear mismatch
    if len(input_entities) == 1 and len(matched_entities) == 1 and \
       fuzz.token_set_ratio(list(input_entities)[0], list(matched_entities)[0]) < fuzzy_threshold:
        return False # e.g., "Ida Fauziyah" vs "Jokowi" with no other entities

    # Calculate ratio of matched input entities
    input_match_ratio = matched_input_entities_count / len(input_entities)
    
    # If a high enough ratio of input entities are matched, consider it a match
    return input_match_ratio >= required_match_ratio


def normalize_text(text): # (unchanged for stemming/stopwords logic)
    if not text or pd.isna(text):
        return ""
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stop_words.discard('akan') # Keep 'akan' for semantic purposes if it's important
    stemmer = StemmerFactory().create_stemmer()
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    text = normalize_synonyms(text) # Apply synonym normalization
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1] # Remove stopwords and short words
    tokens = [stemmer.stem(w) for w in tokens if len(w) > 3] # Apply stemming, consider adjusting length for proper nouns
    return ' '.join(tokens)

# MODIFIED: extract_entities to potentially be more specific
def extract_entities(text):
    if NLP is None:
        raise RuntimeError("NLP model not initialized. Call set_global_models() first.")
    doc = NLP(text)
    entities = set()
    for ent in doc.ents:
        # Focus on Person, Organization, Geopolitical Entity
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            entities.add(ent.text.lower())
    
    # Explicitly check for political parties, as spaCy might miss some Indonesian-specific ones
    text_lower = text.lower()
    for party in POLITICAL_PARTIES:
        # Use regex to match whole words or common phrases for parties more reliably
        # e.g., ensure "pkb" doesn't match "apkb"
        if re.search(r'\b' + re.escape(party) + r'\b', text_lower):
            entities.add(party) # Add the canonical party name or alias
            
    return entities


def extract_political_parties(text): # (unchanged, as it's separate from general entities)
    found_parties = set()
    text_lower = text.lower()
    temp_words = text_lower.split()
    normalized_temp_words = []
    for word in temp_words:
        normalized_temp_words.append(SYNONYM_DICT.get(word, word))
    processed_text = " ".join(normalized_temp_words)
    for party_name_or_alias in POLITICAL_PARTIES:
        if re.search(r'\b' + re.escape(party_name_or_alias) + r'\b', processed_text):
            found_parties.add(party_name_or_alias)
    canonical_parties = set()
    for found_party in found_parties:
        # Check if the found party is a key in SYNONYM_DICT and its value is a canonical party name
        if found_party in SYNONYM_DICT and SYNONYM_DICT[found_party] in POLITICAL_PARTIES:
            canonical_parties.add(SYNONYM_DICT[found_party])
        # If it's a direct match to a canonical party name, add it
        elif found_party in POLITICAL_PARTIES:
            canonical_parties.add(found_party)
        # Handle cases where the found party might be an alias not explicitly in SYNONYM_DICT but related to a canonical name
        else:
            for canonical in POLITICAL_PARTIES:
                if fuzz.token_set_ratio(found_party, canonical) > 90: # High fuzzy match for aliases
                    canonical_parties.add(canonical)
                    break
    return canonical_parties


def summarize_text_simple(text, num_sentences=2): # (unchanged)
    if not text or pd.isna(text):
        return "Tidak ada rangkuman."
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "Tidak ada rangkuman."
    summary = " ".join(sentences[:num_sentences])
    if summary and summary[-1] not in ['.', '!', '?']:
        summary += "."
    return summary

# Data loading and processing function (MODIFIED for Naive Bayes training)
def load_and_process_data_base(data_path="dataset_cnn_summarized - Copy.xlsx"):
    df = pd.read_excel(data_path)
    synonym_dict_generated = auto_generate_synonyms_base(df)
    SYNONYM_DICT.update(synonym_dict_generated) # Ensure SYNONYM_DICT is updated before normalization
    df['text_norm'] = df['title'].apply(normalize_text)
    df['entities'] = df['title'].apply(lambda x: extract_entities(str(x)))
    df['political_parties'] = df['title'].apply(extract_political_parties)

    # NEW: Train Naive Bayes model
    # Ensure 'label' column exists and is binary (0 for valid, 1 for hoax)
    nb_pipeline = None # Initialize nb_pipeline locally
    if 'label' not in df.columns:
        print("Warning: 'label' column not found in data. Cannot train Naive Bayes.")
    else:
        # Features for Naive Bayes: TF-IDF of normalized text
        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1,2))),
            ('nb', MultinomialNB())
        ])
        
        # Filter out rows where text_norm is empty or NaN before training
        train_df = df.dropna(subset=['text_norm', 'label'])
        if not train_df.empty:
            # Assuming 'label' column indicates hoax (e.g., 0 for valid, 1 for hoax)
            # Ensure labels are integers
            train_df['label'] = train_df['label'].astype(int)
            
            # Check if both classes are present
            if len(train_df['label'].unique()) < 2:
                print("Warning: Naive Bayes model requires at least two classes (0 and 1) for training. Only one class found.")
                nb_pipeline = None # Cannot train effectively if only one class
            else:
                nb_pipeline.fit(train_df['text_norm'], train_df['label'])
                print("Naive Bayes model trained successfully.")
                # Save the trained pipeline for later loading if needed for persistence
                joblib.dump(nb_pipeline, 'naive_bayes_pipeline.pkl')
        else:
            print("No valid data for training Naive Bayes model.")
            nb_pipeline = None

    set_global_nb_pipeline(nb_pipeline) # Set the global NB pipeline
    return df, synonym_dict_generated

def auto_generate_synonyms_base(_df): # (unchanged)
    try:
        auto_synonyms = build_enhanced_synonym_dict_base().copy()
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
        print(f"Error generating auto synonyms: {e}")
        return build_enhanced_synonym_dict_base()

def is_suspicious_change(input_text, matched_text): # (slightly refined)
    input_processed_for_diff = normalize_text(input_text)
    matched_processed_for_diff = normalize_text(matched_text)
    
    input_words = input_processed_for_diff.split()
    matched_words = matched_processed_for_diff.split()
    
    matcher = SequenceMatcher(None, input_words, matched_words)
    ratio = matcher.ratio() # measures overall similarity
    
    # Calculate difference in unique words (more sensitive to additions/deletions)
    unique_input_words = set(input_words)
    unique_matched_words = set(matched_words)
    
    added_words = unique_input_words - unique_matched_words
    removed_words = unique_matched_words - unique_input_words
    
    total_unique_words = len(unique_input_words.union(unique_matched_words))
    diff_unique_words_count = len(added_words) + len(removed_words)
    
    prop_diff = diff_unique_words_count / total_unique_words if total_unique_words > 0 else 0
    
    # Criteria for suspicious:
    # 1. Overall similarity is not extremely high (e.g., below 0.98)
    # 2. A significant proportion of unique words have changed (e.g., more than 15-20%)
    # This makes it more sensitive to specific word changes, like names.
    if ratio < 0.98 and prop_diff > 0.15: # Adjust 0.15 (15%) as needed
        return True, prop_diff
    
    return False, prop_diff