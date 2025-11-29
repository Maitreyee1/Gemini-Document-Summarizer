import os
import io
import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables (for local dev)
load_dotenv()

# Get API key
#GEMINI_API_KEY = os.getenv("gemapikey")
GEMINI_API_KEY = st.secrets["gemapikey"]

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# ------------- Helper functions ------------- #

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def summarize_text(text: str, style: str, length: str) -> str:
    system_prompt = (
        "You are a helpful assistant that summarizes documents for end-users.\n"
        "Return a clear, structured summary.\n"
    )

    style_instruction = {
        "Bullet points": "Use bullet points with short, clear sentences.",
        "Paragraph": "Use 2–4 concise paragraphs.",
        "Executive brief": "Write an executive-style brief suitable for managers.",
    }.get(style, "Use bullet points with short, clear sentences.")

    length_instruction = {
        "Very short": "Keep it under 5 bullet points or 100 words.",
        "Short": "About 150–250 words.",
        "Medium": "About 250–400 words.",
        "Detailed": "Up to ~600 words, but stay concise.",
    }.get(length, "About 150–250 words.")

    prompt = f"""{system_prompt}
Summarization style: {style_instruction}
Length: {length_instruction}

Document:
{text}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )

    return response.text

# ------------- Streamlit UI ------------- #

st.set_page_config(page_title="AI Document Summarizer", page_icon="🧠", layout="wide")

st.title("🧠 AI Document Summarizer")
st.caption("Powered by Google Gemini & Streamlit – paste text or upload a document to get a clean summary.")

# Sidebar options
st.sidebar.header("⚙️ Settings")
style = st.sidebar.selectbox(
    "Summary style",
    ["Bullet points", "Paragraph", "Executive brief"],
    index=0,
)
length = st.sidebar.selectbox(
    "Summary length",
    ["Very short", "Short", "Medium", "Detailed"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** gemini-1.5-flash")

# Input area
tab1, tab2 = st.tabs(["✏️ Paste text", "📄 Upload file"])

input_text = ""

with tab1:
    input_text = st.text_area(
        "Paste your document text here:",
        height=250,
        placeholder="Paste an article, research paper text, or any document content...",
    )

with tab2:
    uploaded_file = st.file_uploader("Upload a file (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            input_text = extract_text_from_pdf(file_bytes)
        elif uploaded_file.type in ["text/plain"]:
            input_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if input_text:
            st.success(f"Loaded content from: {uploaded_file.name[:60]}")
            st.text_area("Extracted text (you can edit this before summarizing):",
                         value=input_text,
                         height=250,
                         key="file_text_area")

# Show basic stats
if input_text:
    num_chars = len(input_text)
    num_words = len(input_text.split())
    st.markdown(f"**Document length:** {num_words} words ({num_chars} characters)")

# Summarize button
if st.button("Summarize 🚀"):
    if not input_text or input_text.strip() == "":
        st.warning("Please paste some text or upload a file first.")
    else:
        with st.spinner("Generating summary with Gemini..."):
            try:
                summary = summarize_text(input_text, style=style, length=length)
                st.subheader("📌 Summary")
                st.write(summary)

                # Option to download summary
                st.download_button(
                    label="Download summary as TXT",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"Error while summarizing: {e}")
