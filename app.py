import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="TermsInShort",
    layout="wide"
)

# ==============================
# REMOVE STREAMLIT DEFAULT UI
# ==============================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================
# GLOBAL THEME
# ==============================
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #2f7cf4;
}

/* Main container width */
.block-container {
    padding-top: 1.5rem;
    max-width: 1100px;
}

/* Brand */
.brand {
    font-size: 48px;
    font-weight: 700;
    color: white;
    margin-left: -240px;
    margin-top: -45px;
}

/* Subtitle */
.subtitle {
    color: #E0E7FF;
    margin-top: -6px;
    margin-left: -240px;
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    background-color: #1E40AF;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #1E3A8A;
}

/* Result card */
.result-card {
    background-color: #1E3A8A;
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TOP LEFT BRAND
# ==============================
st.markdown("""
<div class="brand">TermsInShort</div>
<div class="subtitle">Terms of Service (ToS) Scanner</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ==============================
# TEXT EXTRACTION
# ==============================
def fetch_text_from_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text)


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return re.sub(r"\s+", " ", text)

# ==============================
# NLP PIPELINE
# ==============================
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def summarize_text(text, top_n=5):
    sentences = split_sentences(text)

    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf)
    scores = similarity_matrix.sum(axis=1)

    ranked_idx = np.argsort(scores)[::-1]
    return [sentences[i] for i in ranked_idx[:top_n]]


def risk_level(n):
    if n >= 4:
        return "HIGH"
    elif n >= 2:
        return "MEDIUM"
    return "LOW"

# ==============================
# INPUT (NO CARD)
# ==============================
st.subheader("Analyze Terms")

mode = st.radio(
    "Input type",
    ["Website URL", "PDF Document"],
    horizontal=True
)

raw_text = ""

if mode == "Website URL":
    url = st.text_input(
        "Terms of Service or Privacy Policy URL",
        placeholder="https://example.com/terms"
    )
    if url:
        raw_text = fetch_text_from_url(url)
else:
    pdf = st.file_uploader(
        "Upload Terms and Conditions PDF",
        type=["pdf"]
    )
    if pdf:
        raw_text = extract_text_from_pdf(pdf)

# ==============================
# ANALYZE + RESULTS
# ==============================
if st.button("Analyze"):
    if not raw_text.strip():
        st.warning("Please provide a valid URL or PDF.")
    else:
        with st.spinner("Analyzing document..."):
            summary = summarize_text(raw_text)

            html = """
            <div class="result-card">
                <h3>5 Things You Should Know</h3>
                <ol>
            """

            for s in summary:
                html += f"<li>{s}</li>"

            html += f"""
                </ol>
                <p style="margin-top:16px;"><strong>Risk Level:</strong> {risk_level(len(summary))}</p>
                <p style="font-size:12px; opacity:0.85;">
                    This summary is generated using extractive NLP-based sentence ranking.
                </p>
            </div>
            """

            st.markdown(html, unsafe_allow_html=True)

