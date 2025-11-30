"""
Streamlit UI for an AI Document Q&A Assistant (RAG-style).

Features:
- Paste text or upload PDF/TXT.
- Chunk the document and build embeddings.
- Ask natural-language questions about the document.
- Answers are grounded in retrieved chunks (no free hallucinations).

All model + embedding logic lives in services/*.py.
This file focuses on UI and app flow.
"""

import streamlit as st

from services.document_loader import extract_text_from_pdf, chunk_text
from services.rag_qa import build_embeddings_for_chunks, answer_question_over_doc


# =====================================================
# 1. STREAMLIT APP: PAGE CONFIG & TITLE
# =====================================================

st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Q&A Assistant")
st.caption("Upload a document, then ask questions about it using a RAG-style AI assistant.")


# =====================================================
# 2. SESSION STATE INITIALIZATION
# =====================================================

# Store the current document, its chunks, embeddings, and chat history
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)


# =====================================================
# 3. SIDEBAR: SETTINGS
# =====================================================

st.sidebar.header("⚙️ Assistant Settings")

st.sidebar.markdown(
    """
This app uses:

- **Gemini 1.5 Flash** for answering questions  
- **text-embedding-004** for embeddings
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "1. Load a document (paste or upload)\n"
    "2. Wait for processing\n"
    "3. Ask questions in the main panel"
)


# =====================================================
# 4. DOCUMENT INPUT: PASTE OR UPLOAD
# =====================================================

st.subheader("1️⃣ Upload or paste your document")

tab1, tab2 = st.tabs(["✏️ Paste text", "📄 Upload file"])

# ---------- Tab 1: Paste text manually ----------
with tab1:
    pasted_text = st.text_area(
        "Paste your document text here:",
        height=220,
        placeholder="Paste article, report, paper, transcript, etc.",
        value=st.session_state.doc_text,
    )

    if st.button("Use pasted text"):
        if not pasted_text.strip():
            st.warning("Please paste some text before using it.")
        else:
            # Save document text in session state
            st.session_state.doc_text = pasted_text

            # Chunk & embed the document for RAG
            with st.spinner("Processing document (chunking & embedding)..."):
                st.session_state.chunks = chunk_text(st.session_state.doc_text)
                st.session_state.chunk_embeddings = build_embeddings_for_chunks(
                    st.session_state.chunks
                )
                st.session_state.chat_history = []  # reset chat
            st.success("Document updated and processed from pasted text.")


# ---------- Tab 2: Upload file (PDF or TXT) ----------
with tab2:
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
    )

    if uploaded_file is not None:
        # Extract text depending on file type
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            doc_text = extract_text_from_pdf(file_bytes)
        else:  # text/plain
            doc_text = uploaded_file.read().decode("utf-8", errors="ignore")

        # Show extracted text so the user can review/edit it
        st.text_area(
            "Extracted text (you can edit before applying):",
            value=doc_text,
            height=220,
            key="uploaded_text_area",
        )

        if st.button("Use uploaded text"):
            if not st.session_state.uploaded_text_area.strip():
                st.warning("Uploaded text is empty. Check the file content.")
            else:
                st.session_state.doc_text = st.session_state.uploaded_text_area

                # Chunk & embed the document for RAG
                with st.spinner("Processing document (chunking & embedding)..."):
                    st.session_state.chunks = chunk_text(st.session_state.doc_text)
                    st.session_state.chunk_embeddings = build_embeddings_for_chunks(
                        st.session_state.chunks
                    )
                    st.session_state.chat_history = []  # reset chat

                st.success(f"Document loaded and processed from: {uploaded_file.name}")


# =====================================================
# 5. DOCUMENT STATS
# =====================================================

if st.session_state.doc_text:
    num_words = len(st.session_state.doc_text.split())
    num_chars = len(st.session_state.doc_text)
    st.markdown(
        f"📊 **Current document length:** {num_words} words "
        f"({num_chars} characters), {len(st.session_state.chunks)} chunks."
    )
else:
    st.info("No document loaded yet. Paste text or upload a file to get started.")


# =====================================================
# 6. CHAT WITH DOCUMENT (RAG Q&A)
# =====================================================

st.subheader("2️⃣ Ask questions about your document")

# If no document or no embeddings, we can't answer questions
if not st.session_state.doc_text:
    st.warning("Please load a document first to enable Q&A.")
elif not st.session_state.chunks or not st.session_state.chunk_embeddings:
    st.warning(
        "The document has not been processed into chunks/embeddings yet. "
        "Try clicking 'Use pasted text' or 'Use uploaded text' again."
    )
else:
    # Show chat history
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")

    # Input for a new question
    question = st.text_input("Ask a question about the document:")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask_button = st.button("Ask 💬")
    with col2:
        clear_button = st.button("Clear chat 🧹")

    # Clear chat history
    if clear_button:
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Handle question
    if ask_button and question.strip():
        # Add user question to chat history
        st.session_state.chat_history.append(("user", question))

        with st.spinner("Thinking with document context..."):
            try:
                answer = answer_question_over_doc(
                    question,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    st.session_state.chat_history,
                )

                # Add assistant answer to chat history
                st.session_state.chat_history.append(("assistant", answer))

                # Rerun to show new messages in order
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error during Q&A: {e}")
