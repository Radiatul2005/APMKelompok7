import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import util
import joblib # To load the trained Naive Bayes model

# Import core logic functions from the new utils.py file
from utils import (
    load_models_base,
    load_and_process_data_base,
    set_global_models,
    set_global_nb_pipeline, # Import the new setter function
    normalize_text,
    extract_entities,
    extract_political_parties,
    is_meaning_paraphrase,
    is_suspicious_change,
    summarize_text_simple,
    SYNONYM_DICT,
    compare_entities,
    NB_MODEL_PIPELINE # Directly import the global variable for read access in functions
)

# Streamlit page configuration (unchanged)
st.set_page_config(
    page_title="Deteksi Berita Hoax",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
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
        box_shadow: 0 4px 8px rgba(0,0,0,0.1);
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


# Load models and data
@st.cache_resource
def get_cached_models():
    bert_model, nlp = load_models_base()
    # Try loading Naive Bayes pipeline, if it doesn't exist, it will be trained by load_and_process_data_base
    nb_pipeline = None
    try:
        nb_pipeline = joblib.load('naive_bayes_pipeline.pkl')
        st.success("Loaded Naive Bayes pipeline from file.")
    except FileNotFoundError:
        st.warning("Naive Bayes pipeline not found. It will be trained upon data loading (first run might be slower).")
    return bert_model, nlp, nb_pipeline

bert_model, nlp, loaded_nb_pipeline = get_cached_models() # Get the loaded pipeline
set_global_models(bert_model, nlp)
# Set the global NB pipeline in utils.py. This is important to ensure utils.py functions can access it.
set_global_nb_pipeline(loaded_nb_pipeline) 

@st.cache_data
def get_cached_data():
    # This function call will trigger the training of the NB model if it wasn't loaded from file
    df, synonym_dict_generated = load_and_process_data_base() 
    SYNONYM_DICT.update(synonym_dict_generated)
    return df

df = get_cached_data()


@st.cache_data(show_spinner="Generating embeddings...")
def get_bert_embeddings(dataframe):
    if 'text_norm' not in dataframe.columns:
        dataframe['text_norm'] = dataframe['title'].apply(normalize_text)
    return bert_model.encode(dataframe['text_norm'].tolist(), convert_to_tensor=True)

bert_embeddings = get_bert_embeddings(df)

st.sidebar.markdown(f"**Loaded {len(SYNONYM_DICT)} synonym mappings**")


def process_batch_news(news_list, threshold_valid=0.7, nb_hoax_prob_threshold=0.5):
    # Declare NB_MODEL_PIPELINE as global if you intent to use it directly like this
    # global NB_MODEL_PIPELINE # Not strictly needed if only reading, but can prevent NameError if scope is ambiguous.

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

            party_mismatch = False
            # Check for party mismatch more robustly
            if input_parties and parties_mirip:
                if not input_parties.issubset(parties_mirip) and not parties_mirip.issubset(input_parties):
                    party_mismatch = True
            elif input_parties and not parties_mirip: # If input has parties but matched does not, and similarity is high, might be an issue
                party_mismatch = True

            suspicious, prop_diff = is_suspicious_change(judul, berita_mirip['title'])
            
            entity_match_result = compare_entities(input_entities, entities_mirip)

            # Naive Bayes Prediction
            nb_prediction_hoax = False
            nb_hoax_prob = 0.0
            # Access the global NB_MODEL_PIPELINE from utils
            if NB_MODEL_PIPELINE: # Check if the pipeline was loaded/trained successfully
                nb_hoax_prob = NB_MODEL_PIPELINE.predict_proba([judul_norm])[0][1] # Probability of class 1 (hoax)
                if nb_hoax_prob >= nb_hoax_prob_threshold:
                    nb_prediction_hoax = True

            # Consolidated Logic for result
            result = "HOAX" # Default to hoax
            confidence = "UNKNOWN"
            reasons = []

            # Determine specific reasons for HOAX classification
            if not (max_score >= threshold_valid or is_meaning_paraphrase(judul, berita_mirip['title'])):
                reasons.append(f"similarity rendah ({max_score:.2f})")
            if not entity_match_result:
                reasons.append("entitas berbeda")
            if party_mismatch:
                reasons.append("perbedaan partai politik")
            if suspicious:
                reasons.append(f"modifikasi struktur mencurigakan ({prop_diff*100:.1f}% diff)")
            if nb_prediction_hoax:
                reasons.append(f"Sistem mengindikasikan hoaks (prob: {nb_hoax_prob:.2f})")
            
            # Refine reasons if it's a paraphrase but still hoax due to entity/party mismatch
            if is_meaning_paraphrase(judul, berita_mirip['title']) and (not entity_match_result or party_mismatch):
                if "entitas berbeda" in reasons and "paraphrase namun entitas/partai berbeda" not in reasons:
                    reasons.remove("entitas berbeda")
                if "perbedaan partai politik" in reasons and "paraphrase namun entitas/partai berbeda" not in reasons:
                    reasons.remove("perbedaan partai politik")
                if "paraphrase namun entitas/partai berbeda" not in reasons:
                    reasons.append("paraphrase namun entitas/partai berbeda")

            # Final Decision
            if not reasons: # If no reasons found, it means it's VALID
                result = "VALID"
                confidence = "HIGH" if max_score > 0.85 else "MEDIUM"
            else:
                confidence = f"Reason: {', '.join(sorted(list(set(reasons))))}" # Sort and unique reasons for clean display


            short_summary = summarize_text_simple(berita_mirip.get('summarized', 'Tidak ada rangkuman'))

            results.append({
                'Input': judul,
                'Result': result,
                'Similarity': max_score,
                'NB_Hoax_Prob': nb_hoax_prob,
                'Confidence/Reason': confidence,
                'Matched_Title': berita_mirip['title'],
                'Matched_URL': f"[Klik untuk baca berita]({berita_mirip.get('url', 'N/A')})",
                'Matched_Summary_ReRangkum': short_summary
            })

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# Plotting functions (unchanged)
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

def prediksi_berita(judul_berita, threshold_valid=0.7, threshold_rekom=0.35, nb_hoax_prob_threshold=0.5):
    # Declare NB_MODEL_PIPELINE as global to access it
    # global NB_MODEL_PIPELINE # Not strictly needed if only reading, but good practice.
    
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
    if input_parties and parties_mirip:
        if not input_parties.issubset(parties_mirip) and not parties_mirip.issubset(input_parties):
            party_mismatch = True
    elif input_parties and not parties_mirip:
        party_mismatch = True

    suspicious, prop_diff = is_suspicious_change(judul_berita, berita_mirip['title'])

    is_para = is_meaning_paraphrase(judul_berita, berita_mirip['title'])
    
    entity_match_result = compare_entities(input_entities, entities_mirip)

    # Naive Bayes Prediction for single input
    nb_prediction_hoax = False
    nb_hoax_prob = 0.0
    if NB_MODEL_PIPELINE: # Access the global NB_MODEL_PIPELINE from utils
        nb_hoax_prob = NB_MODEL_PIPELINE.predict_proba([judul_norm])[0][1] # Probability of class 1 (hoax)
        if nb_hoax_prob >= nb_hoax_prob_threshold:
            nb_prediction_hoax = True
        st.markdown(f"**Naive Bayes Hoax Probability:** `{nb_hoax_prob:.4f}`")
    else:
        st.warning("Naive Bayes model not loaded. Prediction will proceed without it.")


    # Consolidated Logic for result
    result_class = "HOAX" # Default to hoax
    reasons = []

    # Determine specific reasons for HOAX classification
    if not (max_score >= threshold_valid or is_para):
        reasons.append(f"similarity rendah ({max_score:.2f})")
        if is_para: # If it's a paraphrase but similarity is low, it means is_para is not enough
            reasons.append("paraphrase tapi similarity sangat rendah")
    if not entity_match_result:
        reasons.append("entitas berbeda")
    if party_mismatch:
        reasons.append("perbedaan partai politik")
    if suspicious:
        reasons.append(f"modifikasi struktur mencurigakan ({prop_diff*100:.1f}% diff)")
    if nb_prediction_hoax:
        reasons.append(f"Naive Bayes mengindikasikan hoaks (prob: {nb_hoax_prob:.2f})")

    # Refine the 'paraphrase but different entities/parties' reason
    if is_para and (not entity_match_result or party_mismatch):
        # Remove general entity/party mismatch if more specific paraphrase reason applies
        reasons = [r for r in reasons if "entitas berbeda" not in r and "perbedaan partai politik" not in r]
        if "paraphrase namun entitas/partai berbeda" not in reasons: # Avoid adding duplicate
            reasons.append("paraphrase namun entitas/partai berbeda")

    if not reasons: # If no specific reason found, it means it's VALID
        result_class = "VALID"


    if result_class == "VALID":
        st.markdown(f'<div class="valid-result">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-result-icon">✅</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">VALID, Berita ini terverifikasi dan terpercaya</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="hoax-result">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-result-icon">❌</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">HOAX</div>', unsafe_allow_html=True)
        st.markdown(f"<p>Berita tidak terverifikasi ({' dan '.join(sorted(list(set(reasons))))})</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.plotly_chart(
        plot_similarity_chart(similarity, df),
        use_container_width=True
    )

    # Display details based on whether it was a VALID match or a suspicious paraphrase HOAX
    if (result_class == "VALID" or (result_class == "HOAX" and "modifikasi struktur mencurigakan" in " ".join(reasons))):
        short_summary_matched = summarize_text_simple(berita_mirip.get('summarized', 'Tidak ada rangkuman'))

        st.markdown('<div class="detail-card">', unsafe_allow_html=True)
        st.markdown("<h3>🔍 Informasi Detail</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Judul Berita Serupa</b>: {berita_mirip['title']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Rangkuman Berita</b>: {short_summary_matched}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>🌐 <b>Link</b>: {berita_mirip.get('url', 'Tidak ada URL')}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else: # If classified as HOAX for other reasons (e.g., low similarity, entity mismatch, NB hoax)
        df_valid = df[df['label'] == 0].copy()
        if not df_valid.empty:
            valid_embeddings = bert_model.encode(df_valid['text_norm'].tolist(), convert_to_tensor=True)
            sim_valid = util.pytorch_cos_sim(input_vec, valid_embeddings)[0].cpu().numpy()
            max_sim_rekom = np.max(sim_valid)

            if max_sim_rekom >= threshold_rekom:
                idx_valid = np.argmax(sim_valid)
                rekom = df_valid.iloc[idx_valid]

                short_summary_rekom = summarize_text_simple(rekom.get('summarized', 'Tidak ada rangkuman'))

                st.markdown('<div class="detail-card">', unsafe_allow_html=True)
                st.markdown("<h3>🔁 Rekomendasi Berita Valid:</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Judul</b>: {rekom['title']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Rangkuman Isi Berita Rekomendasi</b>: {short_summary_rekom}</p>", unsafe_allow_html=True)
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

# Sidebar and Main content UI remains unchanged.
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

st.markdown('<h1 class="main-header">🕵️‍♂️ TrueLens</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">Smarter News. Safer Minds</p>', unsafe_allow_html=True)

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
                            # You might want to add a metric for average NB Hoax Probability here
                            avg_nb_prob = results_df['NB_Hoax_Prob'].mean()
                            st.metric("Avg NB Hoax Prob", f"{avg_nb_prob:.3f}")

                        st.markdown("**Hasil Detail:**")
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
                    avg_nb_prob = results_df['NB_Hoax_Prob'].mean()
                    st.metric("Avg NB Hoax Prob", f"{avg_nb_prob:.3f}")

                st.markdown("**Hasil Detail:**")
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