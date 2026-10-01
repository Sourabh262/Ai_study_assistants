from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.exceptions import DocumentError
from app.logging_config import logger


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create the project's standard text splitter."""
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
        raw_chunks = splitter.split_text(text)
        chunks = [c.strip() for c in raw_chunks if c.strip()]

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
        logger.error("Failed to split document text: %s", exc)
        raise DocumentError("Failed to create document chunks.") from exc


def split_pages(
    pages: list[tuple[str, int]],
    source_name: str,
) -> list[tuple[str, dict]]:
    """
    Split page-based document text into chunks while preserving source and page metadata.

    Args:
        pages: List of (page_text, page_number) tuples.
        source_name: File name of the source document.

    Returns:
        List of (chunk_text, metadata_dict) tuples.
    """
    if not pages:
        raise DocumentError("Cannot create chunks from empty pages list.")

    splitter = create_text_splitter()
    chunk_data: list[tuple[str, dict]] = []

    for page_text, page_number in pages:
        if not page_text or not page_text.strip():
            continue

        raw_chunks = splitter.split_text(page_text)
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if cleaned:
                meta = {
                    "source": source_name,
                    "page": page_number,
                }
                chunk_data.append((cleaned, meta))

    if not chunk_data:
        raise DocumentError(
            "No usable chunks were created from the document pages."
        )

    logger.info(
        "Created %s chunks with page metadata for '%s'",
        len(chunk_data),
        source_name,
    )

    return chunk_data