from pathlib import Path

from app.exceptions import DocumentError
from app.loaders.pdf_loader import load_pdf_pages
from app.loaders.text_loader import load_text_file
from app.logging_config import logger
from app.rag.chunker import split_pages
from app.rag.embeddings import generate_embeddings
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class DocumentService:
    """
    Handles the complete document-processing pipeline with page metadata preservation.

    TXT/PDF
       ↓
    Text & Page extraction
       ↓
    Chunking with Metadata (source, page)
       ↓
    Embeddings
       ↓
    Vector Store
    """

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.retriever = Retriever(vector_store=self.vector_store)
        self.file_name: str | None = None
        self.file_type: str | None = None
        self.chunk_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready

    def process_file(
        self,
        file_path: str | Path,
        original_file_name: str | None = None,
    ) -> int:
        path = Path(file_path)

        if not path.exists():
            raise DocumentError(f"Document not found: {path}")

        extension = (
            Path(original_file_name).suffix.lower()
            if original_file_name
            else path.suffix.lower()
        )

        display_name = original_file_name or path.name

        logger.info("Processing document: %s", display_name)

        if extension == ".txt":
            raw_text = load_text_file(path)
            pages = [(raw_text, 1)]
        elif extension == ".pdf":
            pages = load_pdf_pages(path)
        else:
            raise DocumentError(
                "Unsupported file type. Only TXT and PDF files are supported."
            )

        chunk_data = split_pages(pages, source_name=display_name)
        texts = [chunk[0] for chunk in chunk_data]
        metadatas = [chunk[1] for chunk in chunk_data]

        embeddings = generate_embeddings(texts)

        self.vector_store.clear()
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.file_name = display_name
        self.file_type = extension.lstrip(".")
        self.chunk_count = len(texts)

        logger.info(
            "Document processed successfully: %s chunks created with metadata.",
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