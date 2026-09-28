from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.exceptions import EmbeddingError
from app.logging_config import logger


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    The model is loaded only once during the application's
    lifetime and reused for subsequent requests.
    """

    try:
        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL,
        )

        model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Embedding model loaded successfully.")

        return model

    except Exception as exc:
        logger.error(
            "Failed to load embedding model '%s': %s",
            EMBEDDING_MODEL,
            exc,
        )

        raise EmbeddingError(
            "Failed to load the embedding model."
        ) from exc


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        texts: List of text chunks.

    Returns:
        List of embedding vectors.
    """

    if not texts:
        raise EmbeddingError(
            "Cannot generate embeddings for empty text."
        )

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        raise EmbeddingError(
            "No valid text was provided for embedding."
        )

    try:
        model = get_embedding_model()

        logger.info(
            "Generating embeddings for %s chunks.",
            len(cleaned_texts),
        )

        embeddings = model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        result = embeddings.tolist()

        logger.info(
            "Generated embeddings successfully. Shape: (%s, %s)",
            len(result),
            len(result[0]) if result else 0,
        )

        return result

    except EmbeddingError:
        raise

    except Exception as exc:
        logger.error(
            "Failed to generate embeddings: %s",
            exc,
        )

        raise EmbeddingError(
            "Failed to generate document embeddings."
        ) from exc


def generate_query_embedding(query: str) -> list[float]:
    """
    Generate an embedding for a single user query.
    """

    if not query or not query.strip():
        raise EmbeddingError(
            "Cannot generate an embedding for an empty query."
        )

    try:
        model = get_embedding_model()

        embedding = model.encode(
            query.strip(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    except Exception as exc:
        logger.error(
            "Failed to generate query embedding: %s",
            exc,
        )

        raise EmbeddingError(
            "Failed to generate query embedding."
        ) from exc