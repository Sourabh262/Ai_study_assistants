from pathlib import Path

from app.exceptions import DocumentError, EmptyDocumentError
from app.logging_config import logger


def load_text_file(file_path: str | Path) -> str:
    """
    Load and return text from a TXT file.

    Args:
        file_path: Path to the text file.

    Returns:
        Extracted text.

    Raises:
        DocumentError: If the file cannot be read.
        EmptyDocumentError: If the file contains no readable text.
    """

    path = Path(file_path)

    try:
        if not path.exists():
            raise DocumentError(
                f"Document not found: {path}"
            )

        logger.info("Loading TXT document: %s", path.name)

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if not text:
            raise EmptyDocumentError(
                f"The document '{path.name}' contains no readable text."
            )

        logger.info(
            "TXT document loaded successfully: %s characters",
            len(text),
        )

        return text

    except EmptyDocumentError:
        raise

    except OSError as exc:
        logger.error(
            "Failed to read TXT document '%s': %s",
            path.name,
            exc,
        )

        raise DocumentError(
            f"Unable to read document: {path.name}"
        ) from exc