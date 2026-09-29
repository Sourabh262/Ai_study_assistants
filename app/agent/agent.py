import re
from dataclasses import dataclass

from app.agent.router import ToolName, route_question
from app.exceptions import AgentError
from app.logging_config import logger
from app.rag.generator import LLMGenerator
from app.rag.retriever import Retriever
from app.tools.calculator import calculate, format_result
from app.tools.document_search import DocumentSearchTool


@dataclass
class AgentResponse:
    answer: str
    tool_used: ToolName
    sources: list[str]


class StudyAssistantAgent:

    def __init__(
        self,
        llm: LLMGenerator,
        retriever: Retriever | None = None,
    ):
        self.llm = llm
        self.retriever = retriever

        self.document_search = (
            DocumentSearchTool(retriever)
            if retriever is not None
            else None
        )

    def run(self, question: str) -> AgentResponse:

        if not question or not question.strip():
            raise AgentError("Question cannot be empty.")

        question = question.strip()

        has_document = (
            self.retriever is not None
            and self.retriever.vector_store.is_ready
        )

        selected_tool = route_question(question, has_document=has_document)

        logger.info(
            "Agent selected tool: %s",
            selected_tool.value,
        )

        try:
            if selected_tool == ToolName.CALCULATOR:
                return self._run_calculator(question)

            if selected_tool == ToolName.DOCUMENT_SEARCH:
                return self._run_document_search(question)

            return self._run_direct_llm(question)

        except AgentError:
            raise

        except Exception as exc:
            logger.error(
                "Agent execution failed: %s",
                exc,
            )
            raise AgentError(
                "The assistant could not process the question."
            ) from exc

    def _run_calculator(self, question: str) -> AgentResponse:

        expression = self._extract_expression(question)

        result = calculate(expression)
        formatted_result = format_result(result)

        return AgentResponse(
            answer=f"The answer is **{formatted_result}**.",
            tool_used=ToolName.CALCULATOR,
            sources=[],
        )

    def _run_document_search(self, question: str) -> AgentResponse:

        if self.document_search is None or not self.retriever.vector_store.is_ready:
            return AgentResponse(
                answer=(
                    "No document is currently loaded. "
                    "Please upload a TXT or PDF document first."
                ),
                tool_used=ToolName.DOCUMENT_SEARCH,
                sources=[],
            )

        result = self.document_search.search(question)

        if not result.found:
            logger.info("No matching document chunks found; falling back to direct LLM.")
            return self._run_direct_llm(question)

        # Check similarity score of retrieved chunks
        max_score = max((chunk.score for chunk in result.chunks), default=0.0)

        # If similarity score is very low, the question is likely unrelated to the document content
        if max_score < 0.20:
            logger.info(
                "Document search max score too low (%.3f); falling back to direct LLM.",
                max_score,
            )
            return self._run_direct_llm(question)

        answer = self.llm.generate_with_context(
            question=question,
            context=result.context,
        )

        unfound_phrases = [
            "couldn't find that specific information",
            "couldn't find that information",
            "not found in the uploaded document",
            "information is not present in the document",
            "document does not mention",
            "not mentioned in the uploaded document",
        ]

        if any(phrase in answer.lower() for phrase in unfound_phrases):
            logger.info("Information not present in document; providing direct LLM answer.")
            return self._run_direct_llm(question)

        sources = [
            f"Source {index}: similarity={chunk.score:.3f}"
            for index, chunk in enumerate(
                result.chunks,
                start=1,
            )
        ]

        return AgentResponse(
            answer=answer,
            tool_used=ToolName.DOCUMENT_SEARCH,
            sources=sources,
        )

    def _run_direct_llm(self, question: str) -> AgentResponse:

        answer = self.llm.generate_direct(question)

        return AgentResponse(
            answer=answer,
            tool_used=ToolName.DIRECT_LLM,
            sources=[],
        )

    @staticmethod
    def _extract_expression(question: str) -> str:

        cleaned = question.strip()

        prefixes = [
            "calculate",
            "compute",
            "solve",
            "evaluate",
        ]

        lower_question = cleaned.lower()

        for prefix in prefixes:
            if lower_question.startswith(prefix):
                expression = cleaned[len(prefix):].strip()

                if expression:
                    return expression

        match = re.search(
            r"(\d+(?:\.\d+)?\s*"
            r"(?:[\+\-\*\/\%]\s*\d+(?:\.\d+)?)+)",
            cleaned,
        )

        if match:
            return match.group(1)

        percent_match = re.search(
            r"(\d+(?:\.\d+)?\s*%\s*of\s*"
            r"\d+(?:\.\d+)?)",
            cleaned,
            flags=re.IGNORECASE,
        )

        if percent_match:
            return percent_match.group(1)

        raise AgentError(
            "Could not identify a valid mathematical expression."
        )