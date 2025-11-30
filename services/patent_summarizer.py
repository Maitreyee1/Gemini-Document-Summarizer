"""
Patent-aware summarization + key point extraction using Google Gemini.

This module is called from the Streamlit UI (app.py) and provides:

- summarize_text(document_text, style, length, model_name)
- extract_key_points_and_actions(document_text, model_name)

It:
- Uses the shared Gemini client defined in services/ai_client.py
- Uses chunking utilities from services/document_loader.py
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from .document_loader import chunk_text
from .ai_client import client


# =====================================================
# 1. MODEL CONFIG / LOW-LEVEL CALL
# =====================================================

# Default model name for Gemini. You can change this to any valid Gemini model.
DEFAULT_MODEL = "gemini-2.0-flash"  # e.g. "gemini-2.0-flash", "gemini-2.0-pro", etc.


def _call_gemini(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Internal helper to send a single prompt string to Gemini and return text.

    Uses the shared `client` from ai_client.py.
    """
    model = model_name or DEFAULT_MODEL

    # New Google Gemini client syntax (google.genai Client)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    # The Python client conveniently exposes `response.text` as the
    # concatenated text of all parts.
    text = getattr(response, "text", None)
    if not text:
        # Fallback: try to be defensive in case the shape changes.
        try:
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            text = getattr(part, "text", "") or ""
        except Exception:
            text = ""

    if not text:
        raise RuntimeError("Gemini response did not contain any text output.")

    return text


# =====================================================
# 2. PATENT-AWARE SUMMARY PROMPTS
# =====================================================

def _map_summary_style(style: str) -> str:
    """
    Map the UI summary_style string into a short instruction
    that shapes the patent summary.
    """
    style_map = {
        "High-level invention overview (non-technical)": (
            "Write for a non-technical audience. Focus on the problem, the core invention idea, and "
            "what it enables, with minimal jargon."
        ),
        "Technical deep-dive (for engineers/R&D)": (
            "Write for an engineering / R&D audience. Emphasize technical architecture, components, "
            "data flows, algorithms, and implementation details."
        ),
        "Legal-style summary (claims-focused)": (
            "Write in a legal-style tone, focusing on the structure and scope of the claims, "
            "possible independent vs dependent claim relationships, and how the description supports them."
        ),
        "Executive brief for business stakeholders": (
            "Write for business / product leaders. Emphasize what the invention enables, potential "
            "use cases, differentiation, and strategic implications."
        ),
    }
    return style_map.get(
        style,
        "Provide a clear and balanced patent-style summary.",
    )


def _map_summary_length(length: str) -> str:
    """
    Map the UI summary_length into a length instruction.
    """
    length_map = {
        "Very short": "Limit yourself to 3–5 bullet points or a single short paragraph.",
        "Short": "Keep it within 2–3 short paragraphs or up to ~10 bullet points.",
        "Medium": "Provide ~4–6 paragraphs or a mix of sections and bullet points.",
        "Detailed": "Provide a detailed brief with multiple sections and bullet points.",
    }
    return length_map.get(length, "Provide ~4–6 paragraphs.")


def _build_patent_summary_prompt(
    text: str,
    style: str,
    length: str,
    chunk_info: str | None = None,
) -> str:
    """
    Build a patent-aware prompt for summarizing a single text chunk.
    """
    style_instr = _map_summary_style(style)
    length_instr = _map_summary_length(length)

    chunk_note = f"\nNote: {chunk_info}\n" if chunk_info else ""

    prompt = f"""
You are an expert patent attorney AND a clear technical writer.

Your task is to read the patent or patent-like text below and produce
a structured summary.

Goals:
- Apply this style: {style_instr}
- Target length: {length_instr}
- Focus on what a reader needs to understand the invention quickly.

When possible, organize the summary with headings such as:
- Invention overview
- Technical field / background
- Core inventive concept
- Key components / architecture
- Method or process steps (if applicable)
- Claims overview (high-level; do NOT restate all claim text)
- Mention of advantages or improvements over prior art (only if clearly stated)

Important constraints:
- Do NOT invent claims, prior art, or legal arguments that are not clearly supported by the text.
- If something seems missing (e.g., claims section, prior art discussion), explicitly say it appears missing.
- Maintain neutral, non-advocacy language.
- This is NOT legal advice.

{chunk_note}
PATENT TEXT STARTS BELOW
------------------------
{text}
------------------------

Now produce the summary.
"""
    return prompt.strip()


# =====================================================
# 3. PUBLIC FUNCTION: summarize_text
# =====================================================

def summarize_text(
    document_text: str,
    style: str = "High-level invention overview (non-technical)",
    length: str = "Medium",
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Summarize a patent or patent-like document in a patent-aware way.

    - Uses chunking for long documents and then combines with a meta-summary.
    - Parameters are wired from the Streamlit sidebar in app.py.
    """
    text = (document_text or "").strip()
    if not text:
        raise ValueError("No document text provided to summarize.")

    # Use a larger max_chars than the default in document_loader.py
    # because patents can be long, but keep overlap to preserve context.
    chunks: List[str] = chunk_text(text, max_chars=8000, overlap=500)

    # Single-chunk case: simple one-shot summary.
    if len(chunks) == 1:
        prompt = _build_patent_summary_prompt(
            chunks[0],
            style=style,
            length=length,
        )
        return _call_gemini(prompt, model_name=model_name)

    # Multi-chunk case: two-stage summarization.
    partial_summaries: List[str] = []

    for idx, ch in enumerate(chunks, start=1):
        chunk_info = f"This is chunk {idx} of {len(chunks)} from the same patent document."
        prompt = _build_patent_summary_prompt(
            ch,
            style=style,
            length="Medium",  # intermediate level for each chunk
            chunk_info=chunk_info,
        )
        summary = _call_gemini(prompt, model_name=model_name)
        partial_summaries.append(f"### Chunk {idx} summary\n{summary}")

    # Combine chunk-level summaries into one final patent summary.
    combined = "\n\n".join(partial_summaries)

    meta_prompt = f"""
You are an expert patent attorney.

You are given summaries of multiple chunks of the SAME patent or patent-like document.
Your task is to synthesize them into ONE cohesive patent summary.

Requirements:
- Remove duplication and resolve any minor inconsistencies.
- Organize the final output in a logical structure with headings, such as:
  - Invention overview
  - Technical field / background
  - Core inventive concept
  - Key components / architecture
  - Method or process steps (if applicable)
  - Claims overview (high-level)
  - Advantages / improvements (only if clearly described)
  - Potential limitations or ambiguities (only if mentioned or obvious from text)
- Apply this style: {_map_summary_style(style)}
- Target length: {_map_summary_length(length)}
- Maintain neutral tone; do NOT add legal conclusions or advice.
- Do NOT introduce new facts or claims not present in the chunk summaries.

Here are the chunk summaries:

{combined}

Now produce the final, single patent summary.
"""
    return _call_gemini(meta_prompt, model_name=model_name)


# =====================================================
# 4. PATENT-AWARE KEY POINTS / ACTIONS
# =====================================================

def _build_keypoints_prompt(text: str, chunk_info: str | None = None) -> str:
    """
    Build a prompt that extracts patent-specific key points and action items.
    """
    chunk_note = f"\nNote: {chunk_info}\n" if chunk_info else ""

    prompt = f"""
You are an expert in patent analysis (but NOT giving legal advice).

Read the patent or patent-like text below and extract structured information.

Produce your output in clear sections with headings and bullet points:

1. High-level invention summary
2. Apparent independent claim(s) (if you can infer them)
   - Paraphrase only; do NOT copy full claim text verbatim.
3. Apparent dependent claim features (grouped logically)
   - Focus on what is being added or narrowed.
4. Key technical features
   - Hardware / software components, data flows, algorithms, etc.
5. Any references to prior art, background problems, or advantages
   - Only if clearly mentioned.
6. Potential risk / ambiguity flags
   - Only describe what seems ambiguous or complex; do NOT give legal advice.
7. Suggested follow-up actions for a reviewer
   - Examples: "Compare claim 1 scope with product X", "Check prior art around feature Y",
     "Clarify ambiguous term Z in claim drafting", etc.
   - These are task suggestions, not legal conclusions.

Important constraints:
- Do NOT invent claims or legal positions that are not reasonably supported by the text.
- If claims section is missing or partial, explicitly say that and do your best with what is there.
- Keep language practical and readable.

{chunk_note}
PATENT TEXT STARTS BELOW
------------------------
{text}
------------------------

Now produce the analysis.
"""
    return prompt.strip()


def extract_key_points_and_actions(
    document_text: str,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Extracts key patent-specific points and potential follow-up actions.

    This is used by the "Claims & key points" mode in the UI.
    """
    text = (document_text or "").strip()
    if not text:
        raise ValueError("No document text provided for key point extraction.")

    chunks: List[str] = chunk_text(text, max_chars=8000, overlap=500)

    # Single-chunk: direct extraction.
    if len(chunks) == 1:
        prompt = _build_keypoints_prompt(chunks[0])
        return _call_gemini(prompt, model_name=model_name)

    # Multi-chunk: per-chunk extraction, then synthesis.
    chunk_analyses: List[str] = []

    for idx, ch in enumerate(chunks, start=1):
        chunk_info = f"This is chunk {idx} of {len(chunks)} from the same patent document."
        prompt = _build_keypoints_prompt(ch, chunk_info=chunk_info)
        analysis = _call_gemini(prompt, model_name=model_name)
        chunk_analyses.append(f"### Chunk {idx} analysis\n{analysis}")

    combined = "\n\n".join(chunk_analyses)

    meta_prompt = f"""
You are an expert in patent analysis.

You are given analyses for multiple chunks of the SAME patent document.
Your job is to merge them into one cohesive set of points.

Requirements:
- Merge overlapping content and remove duplicates.
- Keep the following sections in the final output (even if some are brief):
  1. High-level invention summary
  2. Apparent independent claim(s) (paraphrased)
  3. Apparent dependent claim features (grouped logically)
  4. Key technical features
  5. References to prior art / background / advantages (if mentioned)
  6. Potential risk / ambiguity flags (no legal advice)
  7. Suggested follow-up actions for a reviewer
- When consolidating independent/dependent claims, avoid double-counting.
- Do NOT introduce new claims or legal conclusions outside what appears in the analyses.
- This is not legal advice, only an informational summary.

Here are the chunk-level analyses:

{combined}

Now produce the final consolidated analysis.
"""
    return _call_gemini(meta_prompt, model_name=model_name)
