import os
import io
import numpy as np
import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv


# -------------------------------------------------------
# Load environment variables (for local development)
# Streamlit Cloud will inject secrets automatically.
# -------------------------------------------------------
load_dotenv()


# Retrieve Gemini API key from Streamlit Secrets. 
GEMINI_API_KEY = st.secrets["gemapikey"]

# Prevent app from running without an API key
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
    st.stop()

# Initialize Gemini client with the API key
client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# 2. HELPER FUNCTIONS: FILE HANDLING & CHUNKING
# =====================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.

    Returns:
        A single string with all extracted text.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """
    Splits a long text into overlapping chunks to fit model context and
    make retrieval easier.

    Args:
        text: Full document text.
        max_chars: Maximum characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Simple char-based chunking (you can make it smarter later)
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]

        # Extend to end of sentence if possible
        last_period = chunk.rfind(".")
        if last_period != -1 and end != length:
            end = start + last_period + 1
            chunk = text[start:end]

        chunks.append(chunk.strip())
        start = max(0, end - overlap)

    return [c for c in chunks if c]


# =====================================================
# 3. HELPER FUNCTIONS: GEMINI SUMMARIZATION
# =====================================================

def summarize_text(text: str, style: str, length: str) -> str:
    """
    Summarizes text using the Gemini generative model.

    Args:
        text: Document text to summarize.
        style: Summary style option selected by the user.
        length: Summary length option selected by the user.

    Returns:
        Summary as a string.
    """
    style_instruction = {
        "Bullet points": "Use concise bullet points.",
        "Paragraph": "Write 2–4 balanced paragraphs.",
        "Executive brief": "Use a high-level executive summary for busy stakeholders.",
    }.get(style, "Use concise bullet points.")

    length_instruction = {
        "Very short": "Keep it under 5 bullet points or 100 words.",
        "Short": "Around 150–250 words.",
        "Medium": "Around 250–400 words.",
        "Detailed": "Up to ~600 words while staying focused.",
    }.get(length, "Around 150–250 words.")

    prompt = f"""
You are an expert document summarizer.

Summarization style: {style_instruction}
Length constraint: {length_instruction}

Given the following document, produce a clear, high-quality summary.

Document:
{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def extract_key_points_and_actions(text: str) -> str:
    """
    Extracts key points and action items from the text.

    Args:
        text: Document text.

    Returns:
        A structured list of key points and action items.
    """
    prompt = f"""
You are an AI assistant that extracts key information from documents.

From the following document, do the following:
1. List the main key points.
2. List action items (who should do what, and by when if mentioned).
3. If there are decisions, list them separately.

Use clear headings and bullet points.

Document:
{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# =====================================================
# 4. HELPER FUNCTIONS: EMBEDDINGS & RAG Q&A
# =====================================================

def get_embedding(text: str) -> np.ndarray:
    """
    Gets an embedding vector for a given text using Gemini embeddings.

    Args:
        text: Input text.

    Returns:
        Numpy array with embedding values.
    """
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    # API returns a list of embeddings; we take the first
    return np.array(result.embeddings[0].values, dtype="float32")


def build_embeddings_for_chunks(chunks):
    """
    Builds embeddings for each text chunk.

    Args:
        chunks: List of text chunks.

    Returns:
        List of embedding vectors (numpy arrays).
    """
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_relevant_chunks(question: str, chunks, chunk_embeddings, top_k: int = 4):
    """
    Retrieves top-k most relevant chunks for a user question using cosine similarity.

    Args:
        question: User's question string.
        chunks: List of document text chunks.
        chunk_embeddings: List of embeddings corresponding to chunks.
        top_k: Number of best chunks to return.

    Returns:
        List of (chunk, score) tuples sorted by score descending.
    """
    if not chunks or not chunk_embeddings:
        return []

    q_emb = get_embedding(question)

    scored = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(q_emb, emb)
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def answer_question_over_doc(question: str, chunks, chunk_embeddings, chat_history):
    """
    Answers a question about the document using retrieved chunks as context.

    Args:
        question: User question.
        chunks: List of document chunks.
        chunk_embeddings: List of embeddings for each chunk.
        chat_history: List of (role, content) messages for conversational context.

    Returns:
        The AI's answer string.
    """
    # Retrieve the most relevant chunks
    top_chunks = retrieve_relevant_chunks(question, chunks, chunk_embeddings, top_k=4)

    if not top_chunks:
        context_text = "No document context available."
    else:
        context_pieces = [c for c, _ in top_chunks]
        context_text = "\n\n---\n\n".join(context_pieces)

    # Build a conversation-style prompt
    history_text = ""
    for role, content in chat_history:
        prefix = "User" if role == "user" else "Assistant"
        history_text += f"{prefix}: {content}\n"

    prompt = f"""
You are an AI assistant that helps users understand a document.

You will be given:
1. Retrieved context from the document (may be incomplete).
2. A conversation history.
3. A new user question.

Use the retrieved context as your primary source of truth.
If something is not in the context, be honest about what you don't know.
Do NOT make up facts that are not supported by the context.

Conversation so far:
{history_text}

Document context:
{context_text}

User question:
{question}

Answer the user's question based only on the document context, in a clear and concise way.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# =====================================================
# 5. STREAMLIT APP: LAYOUT & STATE
# =====================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption("Upload a document, then summarize it, extract key points, or chat with it.")

# Initialize session state for document & chat
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)


# =====================================================
# 6. SIDEBAR: MODES & SETTINGS
# =====================================================

st.sidebar.header("⚙️ Assistant Settings")

mode = st.sidebar.radio(
    "Assistant mode",
    ["Summarize document", "Key points & action items", "Chat with document"],
    index=0,
)

summary_style = st.sidebar.selectbox(
    "Summary style",
    ["Bullet points", "Paragraph", "Executive brief"],
    index=0,
)

summary_length = st.sidebar.selectbox(
    "Summary length",
    ["Very short", "Short", "Medium", "Detailed"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** gemini-2.5-flash, **Embeddings:** text-embedding-004")


# =====================================================
# 7. DOCUMENT INPUT: PASTE OR UPLOAD
# =====================================================

st.subheader("1️⃣ Upload or paste your document")

tab1, tab2 = st.tabs(["✏️ Paste text", "📄 Upload file"])

with tab1:
    pasted_text = st.text_area(
        "Paste your document text here:",
        height=220,
        placeholder="Paste article, research paper, meeting notes, etc.",
        value=st.session_state.doc_text,
    )
    if st.button("Use pasted text"):
        st.session_state.doc_text = pasted_text
        st.session_state.chunks = chunk_text(st.session_state.doc_text)
        st.session_state.chunk_embeddings = build_embeddings_for_chunks(
            st.session_state.chunks
        )
        st.session_state.chat_history = []
        st.success("Document updated from pasted text.")

with tab2:
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
    )
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            doc_text = extract_text_from_pdf(file_bytes)
        else:  # text/plain
            doc_text = uploaded_file.read().decode("utf-8", errors="ignore")

        st.text_area(
            "Extracted text (you can edit below before applying):",
            value=doc_text,
            height=220,
            key="uploaded_text_area",
        )

        if st.button("Use uploaded text"):
            st.session_state.doc_text = st.session_state.uploaded_text_area
            st.session_state.chunks = chunk_text(st.session_state.doc_text)
            st.session_state.chunk_embeddings = build_embeddings_for_chunks(
                st.session_state.chunks
            )
            st.session_state.chat_history = []
            st.success(f"Document loaded from: {uploaded_file.name}")


# Show basic stats for current document
if st.session_state.doc_text:
    num_words = len(st.session_state.doc_text.split())
    num_chars = len(st.session_state.doc_text)
    st.markdown(
        f"📊 **Current document length:** {num_words} words ({num_chars} characters), "
        f"{len(st.session_state.chunks)} chunks."
    )
else:
    st.info("No document loaded yet. Paste text or upload a file to get started.")


# =====================================================
# 8. MODE A: SUMMARIZE DOCUMENT
# =====================================================

if mode == "Summarize document":
    st.subheader("2️⃣ Summarize your document")

    if not st.session_state.doc_text:
        st.warning("Please load a document first.")
    else:
        if st.button("Generate summary 🚀"):
            with st.spinner("Summarizing with Gemini..."):
                try:
                    summary = summarize_text(
                        st.session_state.doc_text,
                        style=summary_style,
                        length=summary_length,
                    )
                    st.subheader("📌 Summary")
                    st.write(summary)

                    st.download_button(
                        label="📥 Download summary as TXT",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error while summarizing: {e}")


# =====================================================
# 9. MODE B: KEY POINTS & ACTION ITEMS
# =====================================================

elif mode == "Key points & action items":
    st.subheader("2️⃣ Extract key points & action items")

    if not st.session_state.doc_text:
        st.warning("Please load a document first.")
    else:
        if st.button("Extract key points ✅"):
            with st.spinner("Extracting key points and action items..."):
                try:
                    result = extract_key_points_and_actions(
                        st.session_state.doc_text
                    )
                    st.subheader("📌 Key Points & Action Items")
                    st.write(result)

                    st.download_button(
                        label="📥 Download as TXT",
                        data=result,
                        file_name="key_points_and_actions.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error while extracting: {e}")


# =====================================================
# 10. MODE C: CHAT WITH DOCUMENT (RAG AGENT)
# =====================================================

elif mode == "Chat with document":
    st.subheader("2️⃣ Ask questions about your document")

    if not st.session_state.doc_text:
        st.warning("Please load a document first.")
    elif not st.session_state.chunks or not st.session_state.chunk_embeddings:
        st.warning(
            "The document has not been processed into chunks yet. "
            "Try reloading the document from the tabs above."
        )
    else:
        # Display chat history
        for role, content in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")

        # New question input
        question = st.text_input("Ask a question about the document:")

        col1, col2 = st.columns([1, 1])
        with col1:
            ask_button = st.button("Ask 💬")
        with col2:
            clear_button = st.button("Clear chat 🧹")

        if clear_button:
            st.session_state.chat_history = []
            st.experimental_rerun()

        if ask_button and question.strip():
            # Append user message to history
            st.session_state.chat_history.append(("user", question))

            with st.spinner("Thinking with document context..."):
                try:
                    answer = answer_question_over_doc(
                        question,
                        st.session_state.chunks,
                        st.session_state.chunk_embeddings,
                        st.session_state.chat_history,
                    )
                    # Append assistant answer to history
                    st.session_state.chat_history.append(("assistant", answer))
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error during Q&A: {e}")

