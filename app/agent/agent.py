import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.exceptions import AgentError, AIStudyAssistantError
from app.logging_config import logger
from app.memory.conversation_memory import ConversationMemory
from app.rag.generator import LLMGenerator
from app.rag.prompts import get_agent_system_prompt
from app.rag.retriever import Retriever
from app.tools.document_search import DocumentSearchTool
from app.tools.registry import ToolRegistry, create_default_registry


@dataclass
class AgentResponse:
    """Standardized response from the Study Assistant Agent."""

    answer: str
    tools_used: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""
    session_id: str = ""

    @property
    def tool_used(self) -> str:
        """Compatibility property for legacy UI/test references."""
        if not self.tools_used:
            return "direct_llm"
        if len(self.tools_used) > 1:
            return "multi_tool"
        return self.tools_used[0]


class StudyAssistantAgent:
    """
    Production-grade Agentic RAG assistant.

    Core Principle:
    The LLM dynamically understands user intent and decides whether to answer directly
    or invoke one or more registered tools. The application validates, controls,
    and executes tool calls securely.
    """

    def __init__(
        self,
        llm: LLMGenerator,
        retriever: Retriever | None = None,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.document_search = (
            DocumentSearchTool(retriever) if retriever is not None else None
        )
        self.tool_registry = tool_registry or create_default_registry(
            document_search_tool=self.document_search
        )
        self.max_steps = max_steps

    def update_retriever(self, retriever: Retriever | None) -> None:
        """Update active document retriever when documents are added or cleared."""
        self.retriever = retriever
        self.document_search = (
            DocumentSearchTool(retriever) if retriever is not None else None
        )
        self.tool_registry = create_default_registry(
            document_search_tool=self.document_search
        )
        logger.info(
            "Agent retriever updated. Document search ready: %s",
            bool(retriever and retriever.vector_store.is_ready),
        )

    def run(
        self,
        question: str,
        memory: ConversationMemory | None = None,
        session_id: str | None = None,
    ) -> AgentResponse:
        """
        Execute the agentic reasoning and tool execution loop.

        Flow:
        User Query + History -> LLM Agent -> Direct Answer OR Tool Calls
        -> Safe Tool Execution -> Tool Results -> LLM -> Final Grounded Answer
        """
        if not question or not question.strip():
            raise AgentError("Question cannot be empty.")

        cleaned_question = question.strip()
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        active_session_id = (
            session_id
            or (memory.session_id if memory else str(uuid.uuid4()))
        )

        start_time = time.perf_counter()
        logger.info(
            "Agent request started. request_id=%s, session_id=%s, query='%s'",
            request_id,
            active_session_id,
            cleaned_question,
        )

        # Build initial messages: System prompt + Conversation history + Current query
        working_messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_agent_system_prompt()}
        ]

        if memory:
            working_messages.extend(memory.get_messages_for_llm())

        working_messages.append({"role": "user", "content": cleaned_question})

        tools_used: list[str] = []
        collected_sources: list[str] = []

        try:
            step = 0
            final_answer: str | None = None

            while step < self.max_steps:
                step += 1
                logger.info(
                    "Agent step %s/%s for request_id=%s",
                    step,
                    self.max_steps,
                    request_id,
                )

                tool_schemas = self.tool_registry.get_schemas()
                response_message = self.llm.chat_completion(
                    messages=working_messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )

                # Check if the LLM decided to invoke any tools
                tool_calls = getattr(response_message, "tool_calls", None)

                if not tool_calls:
                    # Direct answer produced by LLM without further tool calls
                    final_answer = response_message.content or ""
                    logger.info(
                        "LLM produced direct answer on step %s (request_id=%s)",
                        step,
                        request_id,
                    )
                    break

                # Prepare assistant message containing tool calls for history
                tool_calls_payload = []
                for tc in tool_calls:
                    tool_calls_payload.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

                working_messages.append(
                    {
                        "role": "assistant",
                        "content": response_message.content or None,
                        "tool_calls": tool_calls_payload,
                    }
                )

                # Execute all tools requested by the LLM in this turn
                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    tool_call_id = tc.id

                    tools_used.append(tool_name)
                    logger.info(
                        "Executing tool '%s' for request_id=%s (call_id=%s)",
                        tool_name,
                        request_id,
                        tool_call_id,
                    )

                    tool_result = self.tool_registry.execute(
                        name=tool_name,
                        arguments=tool_args,
                    )

                    # Capture source metadata if document_search was called
                    if (
                        tool_name == "document_search"
                        and self.document_search
                        and self.document_search.last_search_result
                    ):
                        for src in self.document_search.last_search_result.sources:
                            source_label = (
                                f"{src['source']} (Page {src['page']}) - "
                                f"Similarity: {src['score']:.2f}"
                            )
                            if source_label not in collected_sources:
                                collected_sources.append(source_label)

                    # Append tool result to working messages for the LLM
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_result.output,
                        }
                    )

            # If loop finished without direct answer (max steps reached), do a final summarization
            if final_answer is None:
                logger.warning(
                    "Agent reached max steps (%s). Requesting final synthesis.",
                    self.max_steps,
                )
                final_completion = self.llm.chat_completion(
                    messages=working_messages,
                    tools=None,
                )
                final_answer = final_completion.content or ""

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Deduplicate tools used while preserving order
            unique_tools = list(dict.fromkeys(tools_used))

            # Update session conversation memory
            if memory:
                memory.add_user_message(cleaned_question)
                memory.add_assistant_message(
                    content=final_answer,
                    sources=collected_sources,
                    tools_used=unique_tools,
                )

            logger.info(
                "Agent request %s completed successfully in %.2f ms. Tools used: %s",
                request_id,
                latency_ms,
                unique_tools,
            )

            return AgentResponse(
                answer=final_answer,
                tools_used=unique_tools,
                sources=collected_sources,
                latency_ms=latency_ms,
                request_id=request_id,
                session_id=active_session_id,
            )

        except AIStudyAssistantError:
            raise

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Agent execution failed for request_id=%s: %s",
                request_id,
                exc,
            )
            raise AgentError(
                "The study assistant could not process the question."
            ) from exc