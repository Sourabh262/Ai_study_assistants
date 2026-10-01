import re
from dataclasses import dataclass, field
from typing import Any

from app.config import MIN_SIMILARITY, TOP_K
from app.exceptions import DocumentSearchError
from app.logging_config import logger
from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import SearchResult, VectorStore


def clean_search_query(query: str) -> str:
    """Strip conversational filler like 'according to the document' before embedding."""
    fillers = [
        r"(?i)\baccording to (the|my) (document|file|resume|cv|pdf|book)\b",
        r"(?i)\b(in|from|of) (the )?(document|file|resume|cv|pdf|book)\b",
        r"(?i)\buploaded (document|file|resume|cv|pdf|book)\b",
        r"(?i)\bwhich i (have )?(attached|uploaded)\b",
        r"(?i)\bi have uploaded (my )?(resume|document|cv|file|pdf|book)?\b",
        r"(?i)\baccording to my (resume|cv|document|file)\b",
        r"(?i)\b\w+\.(txt|pdf)\b",
    ]
    cleaned = query
    for f in fillers:
        cleaned = re.sub(f, "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query


@dataclass
class RetrievedChunk:
    """A document chunk returned by the retriever with metadata."""

    text: str
    score: float
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """
    Handles semantic retrieval from the vector store with metadata preservation.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User's question or search query.
            top_k: Optional override for number of chunks to return.

        Returns:
            Ranked list of retrieved chunks with metadata.
        """
        if not query or not query.strip():
            raise DocumentSearchError("Search query cannot be empty.")

        if not self.vector_store.is_ready:
            raise DocumentSearchError(
                "No document is available for search. Please upload a document first."
            )

        k = top_k if top_k is not None and top_k > 0 else self.top_k

        try:
            cleaned_query = clean_search_query(query.strip())
            logger.info(
                "Retrieving document context for query: '%s' (cleaned: '%s', top_k=%s)",
                query.strip(),
                cleaned_query,
                k,
            )

            query_embedding = generate_query_embedding(cleaned_query)

            results: list[SearchResult] = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                min_similarity=self.min_similarity,
            )

            retrieved_chunks = [
                RetrievedChunk(
                    text=result.text,
                    score=result.score,
                    index=result.index,
                    metadata=result.metadata,
                )
                for result in results
            ]

            logger.info(
                "Retrieved %s relevant chunks.",
                len(retrieved_chunks),
            )

            return retrieved_chunks

        except DocumentSearchError:
            raise
        except Exception as exc:
            logger.error("Document retrieval failed: %s", exc)
            raise DocumentSearchError(
                "Failed to retrieve relevant document content."
            ) from exc


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into structured context for the LLM,
    including file name, page number, and similarity score.
    """
    if not chunks:
        return ""

    context_parts: list[str] = []

    for position, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "Document")
        page = chunk.metadata.get("page", 1)
        score_str = f"{chunk.score:.2f}"
        header = f"[Source {position} | File: {source} | Page: {page} | Similarity: {score_str}]"
        context_parts.append(f"{header}\n{chunk.text}")

    return "\n\n".join(context_parts)