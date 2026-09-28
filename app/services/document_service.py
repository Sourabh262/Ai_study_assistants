from pathlib import Path

from app.exceptions import DocumentError
from app.loaders.pdf_loader import load_pdf_file
from app.loaders.text_loader import load_text_file
from app.logging_config import logger
from app.rag.chunker import split_text
from app.rag.embeddings import generate_embeddings
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class DocumentService:
    """
    Handles the complete document-processing pipeline.

    TXT/PDF
       ↓
    Text extraction
       ↓
    Chunking
       ↓
    Embeddings
       ↓
    Vector Store
    """

    def __init__(self):
        self.vector_store = VectorStore()

        self.retriever = Retriever(
            vector_store=self.vector_store
        )

        self.file_name: str | None = None
        self.file_type: str | None = None
        self.chunk_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready

    def process_file(self, file_path: str | Path) -> int:
        path = Path(file_path)

        if not path.exists():
            raise DocumentError(
                f"Document not found: {path}"
            )

        extension = path.suffix.lower()

        logger.info(
            "Processing document: %s",
            path.name,
        )

        if extension == ".txt":
            text = load_text_file(path)

        elif extension == ".pdf":
            text = load_pdf_file(path)

        else:
            raise DocumentError(
                "Unsupported file type. "
                "Only TXT and PDF files are supported."
            )

        chunks = split_text(text)

        embeddings = generate_embeddings(chunks)

        self.vector_store.clear()

        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
        )

        self.file_name = path.name
        self.file_type = extension.lstrip(".")
        self.chunk_count = len(chunks)

        logger.info(
            "Document processed successfully: "
            "%s chunks created.",
            self.chunk_count,
        )

        return self.chunk_count

    def clear(self) -> None:
        """Clear the currently loaded document."""

        self.vector_store.clear()

        self.file_name = None
        self.file_type = None
        self.chunk_count = 0

        logger.info("Document service cleared.")