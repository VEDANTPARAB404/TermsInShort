import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="TermsInShort",
    page_icon="📄",
    layout="centered"
)

st.title("📄 TermsInShort")
st.caption("Terms & Privacy. In plain English.")

# ==============================
# LOAD HF TOKEN SAFELY
# ==============================
HF_TOKEN = ""
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = ""

# ==============================
# LLM CLIENT (FAST & RELIABLE)
# ==============================
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN
)

# ==============================
# NLP HELPERS
# ==============================
def fetch_text_from_url(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

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


def chunk_text(text, max_chars=2500):
    chunks = []
    while len(text) > max_chars:
        split = text[:max_chars].rfind(".")
        split = split if split != -1 else max_chars
        chunks.append(text[:split])
        text = text[split:]
    chunks.append(text)
    return chunks


# ==============================
# LLM CALL (SAFE)
# ==============================
def llm_summarize(text):
    prompt = f"""
You are a consumer rights assistant.

Analyze the following Terms of Service or Privacy Policy.
List EXACTLY 5 things a user should know.
Highlight risks clearly.
Use simple, non-legal language.
Do NOT defend the company.

TEXT:
{text}

Return only 5 numbered points.
"""

    try:
        return client.text_generation(
            prompt,
            max_new_tokens=250,
            temperature=0.2,
            timeout=60
        )
    except Exception:
        return (
            "⚠️ The AI model is currently busy.\n"
            "1. The policy contains legal limitations.\n"
            "2. User rights may be restricted.\n"
            "3. Data handling terms apply.\n"
            "4. Account actions are controlled by the company.\n"
            "5. Policy terms may change over time."
        )


def calculate_risk(summary):
    keywords = [
        "data", "share", "sell", "terminate",
        "arbitration", "license", "tracking"
    ]
    score = sum(k in summary.lower() for k in keywords)

    if score >= 4:
        return "HIGH 🔴"
    elif score >= 2:
        return "MEDIUM 🟠"
    return "LOW 🟢"


# ==============================
# UI INPUT
# ==============================
mode = st.radio(
    "Choose input type",
    ["Website URL", "PDF Document"],
    key="input_mode"
)

raw_text = ""

if mode == "Website URL":
    url = st.text_input(
        "Paste Terms of Service / Privacy Policy URL",
        key="url_input"
    )
    if url:
        raw_text = fetch_text_from_url(url)

else:
    pdf = st.file_uploader(
        "Upload Terms PDF",
        type=["pdf"],
        key="pdf_input"
    )
    if pdf:
        raw_text = extract_text_from_pdf(pdf)


# ==============================
# ANALYZE
# ==============================
if st.button("Analyze", key="analyze_btn"):
    if not raw_text.strip():
        st.warning("Please provide a valid URL or PDF.")
    else:
        with st.spinner("Analyzing document (may take up to 1 minute)..."):
            chunks = chunk_text(raw_text)

            # 🔒 LIMIT TO FIRST CHUNK (IMPORTANT)
            summary = llm_summarize(chunks[0])

            st.subheader("🔍 5 Things You Should Know")
            st.write(summary)

            st.markdown("---")
            st.write(f"**Risk Level:** {calculate_risk(summary)}")
