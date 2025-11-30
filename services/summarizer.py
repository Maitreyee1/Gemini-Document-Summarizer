from services.ai_client import client


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
    Extracts key points, action items, and decisions from the text.

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
3. List any explicit decisions separately.

Use clear headings and bullet points.

Document:
{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text
