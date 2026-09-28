from dataclasses import dataclass

from app.exceptions import DocumentSearchError
from app.logging_config import logger
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.retriever import format_retrieved_context


@dataclass
class DocumentSearchResult:
    query: str
    chunks: list[RetrievedChunk]
    context: str

    @property
    def found(self) -> bool:
        return bool(self.chunks)


class DocumentSearchTool:
    """Search the currently loaded document using the RAG retriever."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def search(self, query: str) -> DocumentSearchResult:
        if not query or not query.strip():
            raise DocumentSearchError(
                "Document search query cannot be empty."
            )

        try:
            chunks = self.retriever.retrieve(query.strip())
            context = format_retrieved_context(chunks)

            logger.info(
                "Document search completed. Query='%s', results=%s",
                query,
                len(chunks),
            )

            return DocumentSearchResult(
                query=query.strip(),
                chunks=chunks,
                context=context,
            )

        except DocumentSearchError:
            raise

        except Exception as exc:
            logger.error(
                "Document search failed for '%s': %s",
                query,
                exc,
            )
            raise DocumentSearchError(
                "Failed to search the uploaded document."
            ) from exc