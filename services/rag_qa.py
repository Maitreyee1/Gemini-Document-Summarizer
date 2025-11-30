from typing import List, Tuple
import numpy as np

from services.ai_client import client


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
    return np.array(result.embeddings[0].values, dtype="float32")


def build_embeddings_for_chunks(chunks: List[str]) -> List[np.ndarray]:
    """
    Builds embeddings for each text chunk.

    Args:
        chunks: List of text chunks.

    Returns:
        List of embedding vectors (numpy arrays).
    """
    embeddings: List[np.ndarray] = []
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


def retrieve_relevant_chunks(
    question: str,
    chunks: List[str],
    chunk_embeddings: List[np.ndarray],
    top_k: int = 4,
) -> List[Tuple[str, float]]:
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

    scored: List[Tuple[str, float]] = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(q_emb, emb)
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def answer_question_over_doc(
    question: str,
    chunks: List[str],
    chunk_embeddings: List[np.ndarray],
    chat_history: List[tuple],
) -> str:
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
    top_chunks = retrieve_relevant_chunks(
        question, chunks, chunk_embeddings, top_k=4
    )

    if not top_chunks:
        context_text = "No document context available."
    else:
        context_pieces = [c for c, _ in top_chunks]
        context_text = "\n\n---\n\n".join(context_pieces)

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
        model="gemini-1.5-flash",
        contents=prompt,
    )

    return response.text
