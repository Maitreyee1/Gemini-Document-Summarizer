"""
Streamlit UI for an AI Document Summarizer.

Features:
- Paste text or upload PDF/TXT.
- Summarize the document (configurable style & length).
- Extract key points and action items.

Gemini client + model calls live in services/*.py
so this file focuses only on user interface & app flow.
"""

import streamlit as st

from services.document_loader import extract_text_from_pdf
from services.summarizer import summarize_text, extract_key_points_and_actions


# =====================================================
# 1. STREAMLIT APP: PAGE CONFIG & TITLE
# =====================================================

st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Document Summarizer")
st.caption("Upload a document or paste text, then generate summaries or key points using Google Gemini.")


# =====================================================
# 2. SESSION STATE INITIALIZATION
# =====================================================

# We keep the current document text in session state
# so it's shared across tabs and interactions.
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""


# =====================================================
# 3. SIDEBAR: MODES & SETTINGS
# =====================================================

st.sidebar.header("⚙️ Assistant Settings")

# Two modes only: summarization + key points/action items
mode = st.sidebar.radio(
    "What do you want to do?",
    ["Summarize document", "Key points & action items"],
    index=0,
)

# Summary configuration (only used when in "Summarize" mode)
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
st.sidebar.markdown("**Model:** gemini-2.5-flash")


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
        placeholder="Paste article, report, paper, meeting notes, etc.",
        value=st.session_state.doc_text,
    )

    # Button to set the pasted text as the active document
    if st.button("Use pasted text"):
        st.session_state.doc_text = pasted_text
        st.success("Document updated from pasted text.")


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

        # Button to set uploaded text as active document
        if st.button("Use uploaded text"):
            st.session_state.doc_text = st.session_state.uploaded_text_area
            st.success(f"Document loaded from: {uploaded_file.name}")


# =====================================================
# 5. DOCUMENT STATS
# =====================================================

if st.session_state.doc_text:
    num_words = len(st.session_state.doc_text.split())
    num_chars = len(st.session_state.doc_text)
    st.markdown(
        f"📊 **Current document length:** {num_words} words "
        f"({num_chars} characters)."
    )
else:
    st.info("No document loaded yet. Paste text or upload a file to get started.")


# =====================================================
# 6. MODE A: SUMMARIZE DOCUMENT
# =====================================================

if mode == "Summarize document":
    st.subheader("2️⃣ Summarize your document")

    if not st.session_state.doc_text:
        st.warning("Please load or paste a document first.")
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

                    # Optional: let the user download the summary as text
                    st.download_button(
                        label="📥 Download summary as TXT",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error while summarizing: {e}")


# =====================================================
# 7. MODE B: KEY POINTS & ACTION ITEMS
# =====================================================

elif mode == "Key points & action items":
    st.subheader("2️⃣ Extract key points & action items")

    if not st.session_state.doc_text:
        st.warning("Please load or paste a document first.")
    else:
        if st.button("Extract key points ✅"):
            with st.spinner("Extracting key points and action items..."):
                try:
                    result = extract_key_points_and_actions(
                        st.session_state.doc_text
                    )

                    st.subheader("📌 Key Points & Action Items")
                    st.write(result)

                    # Optional: download as text file
                    st.download_button(
                        label="📥 Download as TXT",
                        data=result,
                        file_name="key_points_and_actions.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error while extracting: {e}")
