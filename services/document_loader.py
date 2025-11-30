import io
from typing import List
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.

    Returns:
        A single string containing all extracted text.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Splits a long text into overlapping chunks to fit model context and
    make retrieval easier for RAG-style Q&A.

    Args:
        text: Full document text.
        max_chars: Maximum characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]

        # Try to end at a sentence boundary when possible
        last_period = chunk.rfind(".")
        if last_period != -1 and end != length:
            end = start + last_period + 1
            chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        # Move window forward with overlap
        start = max(0, end - overlap)

    return chunks
