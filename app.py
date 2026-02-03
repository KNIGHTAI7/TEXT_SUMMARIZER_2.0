import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ----------------------
# 🏗️ CONFIGURATION & SETUP
# ----------------------
st.set_page_config(page_title="AI-Powered Text Summarization 2.0", page_icon="🧠", layout="wide")

@st.cache_resource
def setup():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    # Ensure your .h5 and .pkl files are in the same directory
    model = tf.keras.models.load_model("rnn_domain_model.h5")
    with open("assets.pkl", "rb") as f:
        assets = pickle.load(f)
    return model, assets["tokenizer"], assets["label_encoder"]

model, tokenizer, le = setup()

# ----------------------
# 🛠️ PREPROCESSING PIPELINE
# ----------------------
def clean_noise(text):
    """Removes dates, page numbers, and unwanted text fragments."""
    # Remove Dates (MM/DD/YYYY, YYYY-MM-DD, etc.)
    text = re.sub(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', '', text)
    # Remove Page Numbers (e.g., 'Page 1 of 10', 'p. 45', '[12]')
    text = re.sub(r'Page\s*\d+|p\.\s*\d+|\[\d+\]', '', text, flags=re.IGNORECASE)
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_analytical_title(summary_sentences):
    """Creates a title based on the most significant summary sentence."""
    if not summary_sentences:
        return "NO CONTENT DETECTED"
    # Logic: The first sentence of the TextRank result is the most central idea
    title_base = str(summary_sentences[0])
    title = title_base.split('.')[0] # Take only the first clause
    return title[:85].upper()

# ----------------------
# 🎨 MINIMALIST UI (DARK MODE)
# ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&family=Inter:wght@300;500&display=swap');
    .stApp { background-color: #0B0B0B; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .stTextArea textarea { background-color: #141414 !important; color: #FFFFFF !important; border: 1px solid #222 !important; border-radius: 4px !important; }
    .output-section { border-left: 1px solid #333; padding-left: 25px; margin-top: 40px; }
    .label { font-family: 'JetBrains Mono', monospace; color: #00FF41; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px; }
    .content-text { color: #FFF; margin-bottom: 30px; font-size: 1.1rem; }
    .title-text { font-size: 1.8rem; font-weight: 500; line-height: 1.2; margin-bottom: 30px; }
    .stButton>button { width: 100%; background-color: #FFF !important; color: #000 !important; font-weight: 600 !important; height: 50px; border: none !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# 🕹️ APP INTERFACE
# ----------------------
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown("## 📑AI-Powered Text Summarization 2.0")
    st.caption("Enhanced RNN Sequence Analysis")
    article_input = st.text_area("", placeholder="Paste article content here...", height=450)
    process_btn = st.button("SUMMARIZE✅")

with col2:
    if process_btn and article_input.strip():
        # 1. Cleaning
        cleaned_text = clean_noise(article_input)
        
        # 2. RNN Domain Prediction
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=200)
        pred = model.predict(padded, verbose=0)
        domain = le.inverse_transform([np.argmax(pred)])[0]

        # 3. TextRank Summarization (Strict 5-6 Sentences)
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_nodes = summarizer(parser.document, 6) # Request 6 for 5-6 sentence depth
        summary_text = " ".join(str(s) for s in summary_nodes)
        
        # 4. Analytical Title
        analytical_title = generate_analytical_title(summary_nodes)

        # 🚀 OUTPUT SEQUENCE
        st.markdown("<div class='output-section'>", unsafe_allow_html=True)
        
        # 1. TITLE
        st.markdown("<div class='label'>TITLE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='title-text'>{analytical_title}</div>", unsafe_allow_html=True)

        # 2. DOMAIN
        st.markdown("<div class='label'>DOMAIN</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='content-text' style='color:#00FF41; font-family:\"JetBrains Mono\";'>{domain.upper()}</div>", unsafe_allow_html=True)

        # 3. SUMMARY
        st.markdown("<div class='label'>SUMMARY</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='content-text' style='line-height:1.8; color:#BBB;'>{summary_text}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:150px; color:#333; text-align:center; font-family:\"JetBrains Mono\";'>SYSTEM IDLE: AWAITING INPUT</div>", unsafe_allow_html=True)