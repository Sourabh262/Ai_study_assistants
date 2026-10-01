from pathlib import Path

from pypdf import PdfReader

from app.exceptions import DocumentError, EmptyDocumentError
from app.logging_config import logger


def load_pdf_pages(file_path: str | Path) -> list[tuple[str, int]]:
    """
    Extract text from a PDF document page by page, preserving page numbers.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of tuples: (page_text, page_number) where page_number is 1-indexed.

    Raises:
        DocumentError: If the PDF cannot be read.
        EmptyDocumentError: If no readable text is found.
    """
    path = Path(file_path)

    if not path.exists():
        raise DocumentError(f"Document not found: {path}")

    try:
        logger.info("Loading PDF document pages: %s", path.name)
        reader = PdfReader(str(path))

        if not reader.pages:
            raise EmptyDocumentError(
                f"The PDF '{path.name}' contains no pages."
            )

        pages_data: list[tuple[str, int]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_data.append((page_text.strip(), page_number))
            except Exception as exc:
                logger.warning(
                    "Could not extract text from page %s of '%s': %s",
                    page_number,
                    path.name,
                    exc,
                )

        if not pages_data:
            raise EmptyDocumentError(
                f"The PDF '{path.name}' contains no readable text."
            )

        logger.info(
            "PDF loaded successfully: %s pages with text from %s total pages.",
            len(pages_data),
            len(reader.pages),
        )

        return pages_data

    except EmptyDocumentError:
        raise
    except Exception as exc:
        logger.error("Failed to read PDF '%s': %s", path.name, exc)
        raise DocumentError(
            f"Unable to read PDF document: {path.name}"
        ) from exc


def load_pdf_file(file_path: str | Path) -> str:
    """
    Extract text from a PDF document as a single concatenated string.
    """
    pages_data = load_pdf_pages(file_path)
    return "\n\n".join(text for text, _ in pages_data)