from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.exceptions import DocumentError
from app.logging_config import logger


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the project's standard text splitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def split_text(text: str) -> list[str]:
    """
    Split document text into overlapping chunks.

    Args:
        text: Raw document text.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        raise DocumentError(
            "Cannot create chunks from empty document text."
        )

    try:
        splitter = create_text_splitter()

        chunks = splitter.split_text(text)

        chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

        if not chunks:
            raise DocumentError(
                "No usable chunks were created from the document."
            )

        logger.info(
            "Created %s chunks using chunk_size=%s and chunk_overlap=%s",
            len(chunks),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        return chunks

    except DocumentError:
        raise

    except Exception as exc:
        logger.error(
            "Failed to split document text: %s",
            exc,
        )

        raise DocumentError(
            "Failed to create document chunks."
        ) from exc