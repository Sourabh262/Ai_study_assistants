from dataclasses import dataclass

import numpy as np

from app.config import MIN_SIMILARITY, TOP_K
from app.exceptions import VectorStoreError
from app.logging_config import logger


@dataclass
class SearchResult:
    """Represents one retrieved document chunk."""

    text: str
    score: float
    index: int


class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.

    Stores:
    - Document chunks
    - Their embeddings

    This store is intentionally lightweight for the capstone.
    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None

    @property
    def is_ready(self) -> bool:
        """Return True when the vector store contains documents."""

        return (
            bool(self._texts)
            and self._embeddings is not None
            and len(self._embeddings) > 0
        )

    @property
    def document_count(self) -> int:
        """Return the number of stored chunks."""

        return len(self._texts)

    def clear(self) -> None:
        """Remove all stored documents and embeddings."""

        self._texts = []
        self._embeddings = None

        logger.info("Vector store cleared.")

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add document chunks and their embeddings.

        Args:
            texts: Document chunks.
            embeddings: Corresponding embedding vectors.
        """

        if not texts:
            raise VectorStoreError(
                "Cannot add empty documents to the vector store."
            )

        if not embeddings:
            raise VectorStoreError(
                "Cannot add empty embeddings to the vector store."
            )

        if len(texts) != len(embeddings):
            raise VectorStoreError(
                "Number of texts must match number of embeddings."
            )

        try:
            embedding_array = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            if embedding_array.ndim != 2:
                raise VectorStoreError(
                    "Embeddings must be a 2-dimensional array."
                )

            if embedding_array.shape[0] != len(texts):
                raise VectorStoreError(
                    "Embedding count does not match text count."
                )

            self._texts = [
                text.strip()
                for text in texts
                if text and text.strip()
            ]

            if len(self._texts) != embedding_array.shape[0]:
                raise VectorStoreError(
                    "Some document chunks are empty."
                )

            self._embeddings = embedding_array

            logger.info(
                "Added %s document chunks to vector store.",
                len(self._texts),
            )

        except VectorStoreError:
            raise

        except Exception as exc:
            logger.error(
                "Failed to add documents to vector store: %s",
                exc,
            )

            raise VectorStoreError(
                "Failed to initialize the vector store."
            ) from exc

    @staticmethod
    def _cosine_similarity(
        query_vector: np.ndarray,
        document_vectors: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarity between query and documents."""

        query_norm = np.linalg.norm(query_vector)

        document_norms = np.linalg.norm(
            document_vectors,
            axis=1,
        )

        if query_norm == 0:
            raise VectorStoreError(
                "Query embedding has zero magnitude."
            )

        # Avoid division by zero for malformed document embeddings.
        safe_document_norms = np.where(
            document_norms == 0,
            1e-12,
            document_norms,
        )

        similarities = (
            document_vectors @ query_vector
        ) / (
            safe_document_norms * query_norm
        )

        return similarities

    def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> list[SearchResult]:
        """
        Search for the most relevant document chunks.

        Args:
            query_embedding: Embedding of the user's query.
            top_k: Maximum number of results.
            min_similarity: Minimum cosine similarity.

        Returns:
            Ranked list of SearchResult objects.
        """

        if not self.is_ready:
            raise VectorStoreError(
                "Vector store is empty. Upload and process a document first."
            )

        if not query_embedding:
            raise VectorStoreError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise VectorStoreError(
                "top_k must be greater than zero."
            )

        try:
            query_vector = np.asarray(
                query_embedding,
                dtype=np.float32,
            )

            if query_vector.ndim != 1:
                raise VectorStoreError(
                    "Query embedding must be a 1-dimensional vector."
                )

            if self._embeddings is None:
                raise VectorStoreError(
                    "Vector store embeddings are not initialized."
                )

            if query_vector.shape[0] != self._embeddings.shape[1]:
                raise VectorStoreError(
                    "Query embedding dimension does not match "
                    "document embedding dimension."
                )

            similarities = self._cosine_similarity(
                query_vector,
                self._embeddings,
            )

            ranked_indices = np.argsort(
                similarities
            )[::-1]

            # For small documents (<= 5 chunks like a resume or short notes),
            # provide all chunks to the LLM so no critical information (like names/headers) is missed.
            if len(self._texts) <= 5:
                results = [
                    SearchResult(
                        text=self._texts[idx],
                        score=float(similarities[idx]),
                        index=int(idx),
                    )
                    for idx in ranked_indices
                ]
                logger.info(
                    "Vector search completed for small document. Returning all %s chunks.",
                    len(results),
                )
                return results

            results: list[SearchResult] = []

            for index in ranked_indices[:top_k]:
                score = float(similarities[index])

                if score < min_similarity:
                    continue

                results.append(
                    SearchResult(
                        text=self._texts[index],
                        score=score,
                        index=int(index),
                    )
                )

            # Fallback: if all results were below min_similarity but we have documents,
            # return the top match rather than returning an empty result.
            if not results and len(self._texts) > 0:
                best_idx = int(ranked_indices[0])
                results.append(
                    SearchResult(
                        text=self._texts[best_idx],
                        score=float(similarities[best_idx]),
                        index=best_idx,
                    )
                )

            logger.info(
                "Vector search completed. Retrieved %s results.",
                len(results),
            )

            return results

        except VectorStoreError:
            raise

        except Exception as exc:
            logger.error(
                "Vector search failed: %s",
                exc,
            )

            raise VectorStoreError(
                "Failed to search the vector store."
            ) from exc