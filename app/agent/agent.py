import re
from dataclasses import dataclass

from app.agent.router import ToolName, is_document_query, route_question
from app.exceptions import AgentError
from app.logging_config import logger
from app.rag.generator import LLMGenerator
from app.rag.retriever import Retriever
from app.tools.calculator import calculate, format_result
from app.tools.document_search import DocumentSearchTool


def is_unfound_response(text: str) -> bool:
    """Check if the model indicates the requested information was not found or is absent."""
    lowered = text.lower()
    indicators = [
        "couldn't find",
        "could not find",
        "cannot find",
        "can't find",
        "not found",
        "not mentioned",
        "does not mention",
        "doesn't mention",
        "not present",
        "not provided",
        "does not provide",
        "doesn't provide",
        "does not contain",
        "doesn't contain",
        "don't have any information",
        "do not have any information",
        "don't have information",
        "do not have information",
        "no information about",
        "no information is available",
        "no information available",
        "no mention of",
        "not specified",
        "not stated",
        "not available in the",
        "i don't have your",
        "i do not have your",
        "i don't know your",
        "i do not know your",
        "i don't have access to your",
        "i do not have access to your",
        "unable to find",
        "not in the document",
        "not in the uploaded",
    ]
    return any(phrase in lowered for phrase in indicators)


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

            if selected_tool == ToolName.MULTI_TOOL:
                return self._run_multi_tool(question, has_document=has_document)

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

    def _run_multi_tool(self, question: str, has_document: bool) -> AgentResponse:
        """
        Handle compound questions that involve both mathematical calculation
        and conceptual / document inquiries.
        """
        calc_result_str = None
        try:
            expression = self._extract_expression(question)
            result = calculate(expression)
            formatted_result = format_result(result)
            calc_result_str = f"{expression} = {formatted_result}"
        except Exception as exc:
            logger.warning("Could not pre-calculate expression in multi-tool: %s", exc)

        compound_prompt = f"""The user asked a multi-part question:
"{question}"

{f"The mathematical calculation was solved by the calculator tool: {calc_result_str}" if calc_result_str else ""}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer to the non-mathematical part of the question (such as concepts, definitions, or document inquiries).
2. Clearly state the mathematical calculation result ({calc_result_str if calc_result_str else "as computed"}).
3. Address both parts of the user's inquiry thoroughly.
"""

        sources = []
        if has_document and self.document_search is not None:
            doc_result = self.document_search.search(question)
            max_score = max((chunk.score for chunk in doc_result.chunks), default=0.0)
            if doc_result.found and max_score >= 0.25:
                rag_prompt = f"""DOCUMENT CONTEXT FROM UPLOADED FILE:
-----------------
{doc_result.context}
-----------------

{compound_prompt}
"""
                answer = self.llm._generate(rag_prompt)
                sources = []
                if not is_unfound_response(answer):
                    sources = [
                        f"Source {index}: similarity={chunk.score:.3f}"
                        for index, chunk in enumerate(doc_result.chunks, start=1)
                    ]
                return AgentResponse(
                    answer=answer,
                    tool_used=ToolName.MULTI_TOOL,
                    sources=sources,
                )

        answer = self.llm._generate(compound_prompt)
        return AgentResponse(
            answer=answer,
            tool_used=ToolName.MULTI_TOOL,
            sources=[],
        )

    def _run_document_search(self, question: str) -> AgentResponse:

        if self.document_search is None or not self.retriever.vector_store.is_ready:
            return AgentResponse(
                answer=(
                    "No document is currently loaded. "
                    "Please upload a TXT or PDF document first."
                ),
                tool_used=ToolName.DIRECT_LLM,
                sources=[],
            )

        result = self.document_search.search(question)

        if not result.found:
            logger.info("No matching document chunks found; falling back to direct LLM.")
            return self._run_direct_llm(question)

        # Check similarity score of retrieved chunks
        max_score = max((chunk.score for chunk in result.chunks), default=0.0)
        explicit_doc = is_document_query(question)

        # If similarity score is very low, and it is NOT an explicit document query,
        # the question is a general question unrelated to the document content.
        if max_score < 0.20 and not explicit_doc:
            logger.info(
                "Document search max score too low (%.3f) for non-document query; falling back to direct LLM.",
                max_score,
            )
            return self._run_direct_llm(question)

        answer = self.llm.generate_with_context(
            question=question,
            context=result.context,
        )

        # If the model indicates the information was NOT found or not available in the document:
        if is_unfound_response(answer):
            logger.info("Information not present in document or model has no info.")
            # If the user did not explicitly ask about the document, provide a clean direct LLM answer
            if not explicit_doc:
                return self._run_direct_llm(question)

            # If the user explicitly asked about the document, keep the answer explaining it was not found,
            # but do NOT return sources and attribute to direct_llm.
            return AgentResponse(
                answer=answer,
                tool_used=ToolName.DIRECT_LLM,
                sources=[],
            )

        # Only provide sources and attribute to document_search when the document actually supplied the answer!
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