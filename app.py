"""
Streamlit UI for an AI Patent Summarizer.

Features:
- Paste text or upload PDF/TXT (e.g., patent, application, office action).
- Summarize the patent (configurable style & length).
- Extract key points / claims & potential action items.

Gemini client + model calls live in services/*.py
so this file focuses only on user interface & app flow.
"""

import streamlit as st

from services.document_loader import extract_text_from_pdf
from services.patent_summarizer import summarize_text, extract_key_points_and_actions


# =====================================================
# 1. STREAMLIT APP: PAGE CONFIG & TITLE
# =====================================================

st.set_page_config(
    page_title="AI Patent Summarizer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Patent Summarizer")
st.caption(
    "Upload a patent / patent-like document or paste text, then generate structured patent summaries "
    "or claims/key points using Google Gemini."
)


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

st.sidebar.header("⚙️ Patent Assistant Settings")

# Two modes: patent summary + claims / key points
mode = st.sidebar.radio(
    "What do you want to do?",
    ["Patent summary", "Claims & key points"],
    index=0,
)

# Summary configuration (used when in "Patent summary" mode)
summary_style = st.sidebar.selectbox(
    "Summary style",
    [
        "High-level invention overview (non-technical)",
        "Technical deep-dive (for engineers/R&D)",
        "Legal-style summary (claims-focused)",
        "Executive brief for business stakeholders",
    ],
    index=0,
)

summary_length = st.sidebar.selectbox(
    "Summary length",
    ["Very short", "Short", "Medium", "Detailed"],
    index=2,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Document type:** Patent / patent-like")
st.sidebar.markdown("**Model:** gemini-2.5-flash")


# =====================================================
# 4. DOCUMENT INPUT: PASTE OR UPLOAD
# =====================================================

st.subheader("1️⃣ Upload or paste your patent document")

tab1, tab2 = st.tabs(["✏️ Paste text", "📄 Upload file"])

# ---------- Tab 1: Paste text manually ----------
with tab1:
    pasted_text = st.text_area(
        "Paste your patent text here (claims, description, office action, etc.):",
        height=220,
        placeholder="Paste text from a patent, application, office action, prior art, etc.",
        value=st.session_state.doc_text,
    )

    # Button to set the pasted text as the active document
    if st.button("Use pasted text"):
        st.session_state.doc_text = pasted_text
        st.success("Patent document updated from pasted text.")


# ---------- Tab 2: Upload file (PDF or TXT) ----------
with tab2:
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file (e.g., patent, application, office action)",
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
            st.success(f"Patent document loaded from: {uploaded_file.name}")


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
    st.info(
        "No patent document loaded yet. Paste text or upload a file to get started."
    )


# =====================================================
# 6. MODE A: PATENT SUMMARY
# =====================================================

if mode == "Patent summary":
    st.subheader("2️⃣ Summarize your patent")

    st.markdown(
        "The assistant will aim to produce a structured patent-aware summary.\n\n"
        "- For *High-level overview*, it focuses on the core idea in simple language.\n"
        "- For *Technical deep-dive*, it emphasizes architecture, components, and method steps.\n"
        "- For *Legal-style summary*, it emphasizes claims, scope, and relationships.\n"
    )

    if not st.session_state.doc_text:
        st.warning("Please load or paste a patent document first.")
    else:
        if st.button("Generate patent summary 🚀"):
            with st.spinner("Summarizing patent with Gemini..."):
                try:
                    # NOTE:
                    # `summary_style` is now patent-specific (see options above).
                    # You can adjust the prompt logic inside services/patent_summarizer.py
                    # to interpret these styles and build a patent-aware prompt.
                    summary = summarize_text(
                        st.session_state.doc_text,
                        style=summary_style,
                        length=summary_length,
                    )

                    st.subheader("📌 Patent Summary")
                    st.write(summary)

                    # Optional: let the user download the summary as text
                    st.download_button(
                        label="📥 Download summary as TXT",
                        data=summary,
                        file_name="patent_summary.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error while summarizing patent: {e}")


# =====================================================
# 7. MODE B: CLAIMS & KEY POINTS
# =====================================================

elif mode == "Claims & key points":
    st.subheader("2️⃣ Extract claims, key points & potential actions")

    st.markdown(
        "This mode is useful for quickly seeing:\n"
        "- Main independent/dependent claims (as understood from text)\n"
        "- Key technical features\n"
        "- Potential follow-ups or action items (e.g., for review, responses, or product impact)\n"
    )

    if not st.session_state.doc_text:
        st.warning("Please load or paste a patent document first.")
    else:
        if st.button("Extract claims & key points ✅"):
            with st.spinner("Extracting claims, key points, and action items..."):
                try:
                    result = extract_key_points_and_actions(
                        st.session_state.doc_text
                    )

                    st.subheader("📌 Claims, Key Points & Action Items")
                    st.write(result)

                    # Optional: download as text file
                    st.download_button(
                        label="📥 Download as TXT",
                        data=result,
                        file_name="patent_claims_key_points.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error while extracting from patent: {e}")
