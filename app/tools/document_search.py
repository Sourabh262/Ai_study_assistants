from dataclasses import dataclass, field
from typing import Any

from app.exceptions import DocumentSearchError
from app.logging_config import logger
from app.rag.retriever import RetrievedChunk, Retriever, format_retrieved_context


@dataclass
class DocumentSearchResult:
    query: str
    chunks: list[RetrievedChunk]
    context: str
    sources: list[dict[str, Any]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.chunks)


class DocumentSearchTool:
    """Search the currently loaded document using semantic vector retrieval with page metadata."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever
        self.last_search_result: DocumentSearchResult | None = None

    def search(self, query: str, top_k: int = 5) -> DocumentSearchResult:
        if not query or not query.strip():
            raise DocumentSearchError("Document search query cannot be empty.")

        if self.retriever is None or not self.retriever.vector_store.is_ready:
            logger.info("DocumentSearchTool called but no document is ready.")
            empty_result = DocumentSearchResult(
                query=query.strip(),
                chunks=[],
                context="No document is currently uploaded. Please inform the user to upload a PDF or TXT file first.",
                sources=[],
            )
            self.last_search_result = empty_result
            return empty_result

        try:
            chunks = self.retriever.retrieve(query.strip(), top_k=top_k)
            context = format_retrieved_context(chunks)

            sources: list[dict[str, Any]] = []
            for chunk in chunks:
                sources.append(
                    {
                        "source": chunk.metadata.get("source", "Document"),
                        "page": chunk.metadata.get("page", 1),
                        "score": round(chunk.score, 3),
                        "text": chunk.text,
                    }
                )

            logger.info(
                "Document search completed. Query='%s', results=%s",
                query,
                len(chunks),
            )

            if not chunks:
                context = (
                    f"No relevant information was found in the uploaded document "
                    f"matching the query: '{query.strip()}'."
                )

            result = DocumentSearchResult(
                query=query.strip(),
                chunks=chunks,
                context=context,
                sources=sources,
            )
            self.last_search_result = result
            return result

        except DocumentSearchError:
            raise
        except Exception as exc:
            logger.error("Document search failed for '%s': %s", query, exc)
            raise DocumentSearchError(
                "Failed to search the uploaded document."
            ) from exc

    def execute(self, query: str, top_k: int = 5) -> str:
        """Execute search and return formatted string for LLM tool call response."""
        result = self.search(query=query, top_k=top_k)
        self.last_search_result = result
        return result.context