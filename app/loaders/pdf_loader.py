from pathlib import Path

from pypdf import PdfReader

from app.exceptions import DocumentError, EmptyDocumentError
from app.logging_config import logger


def load_pdf_file(file_path: str | Path) -> str:
    """
    Extract text from a PDF document.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text from all readable pages.

    Raises:
        DocumentError: If the PDF cannot be read.
        EmptyDocumentError: If no readable text is found.
    """

    path = Path(file_path)

    if not path.exists():
        raise DocumentError(
            f"Document not found: {path}"
        )

    try:
        logger.info("Loading PDF document: %s", path.name)

        reader = PdfReader(str(path))

        if not reader.pages:
            raise EmptyDocumentError(
                f"The PDF '{path.name}' contains no pages."
            )

        pages_text: list[str] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                page_text = page.extract_text() or ""

                if page_text.strip():
                    pages_text.append(page_text.strip())

            except Exception as exc:
                logger.warning(
                    "Could not extract text from page %s of '%s': %s",
                    page_number,
                    path.name,
                    exc,
                )

        text = "\n\n".join(pages_text).strip()

        if not text:
            raise EmptyDocumentError(
                f"The PDF '{path.name}' contains no readable text."
            )

        logger.info(
            "PDF loaded successfully: %s pages, %s characters",
            len(reader.pages),
            len(text),
        )

        return text

    except EmptyDocumentError:
        raise

    except Exception as exc:
        logger.error(
            "Failed to read PDF '%s': %s",
            path.name,
            exc,
        )

        raise DocumentError(
            f"Unable to read PDF document: {path.name}"
        ) from exc