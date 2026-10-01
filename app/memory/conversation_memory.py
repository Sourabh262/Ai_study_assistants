import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ChatMessage:
    """Represents a single message in the conversation history."""

    role: str  # "user", "assistant", "tool", "system"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    sources: list[str] | None = None
    tools_used: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert message to dictionary compatible with LLM chat completion API."""
        msg: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            msg["content"] = self.content

        if self.role == "assistant" and self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        if self.role == "tool":
            if self.tool_call_id:
                msg["tool_call_id"] = self.tool_call_id
            if self.name:
                msg["name"] = self.name
            msg["content"] = self.content or ""

        return msg

    def to_display_dict(self) -> dict[str, Any]:
        """Convert message to dictionary compatible with UI display."""
        return {
            "role": self.role,
            "content": self.content or "",
            "sources": self.sources or [],
            "tools_used": self.tools_used or [],
            "timestamp": self.timestamp,
        }


class ConversationMemory:
    """
    In-memory, session-isolated conversation history manager.

    Maintains:
    - Complete multi-turn messages for context
    - Tool calls and results for grounded reasoning
    - Clean display messages for the UI
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self._messages: list[ChatMessage] = []

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def add_user_message(self, content: str) -> ChatMessage:
        """Record a user query."""
        msg = ChatMessage(role="user", content=content.strip())
        self._messages.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        sources: list[str] | None = None,
        tools_used: list[str] | None = None,
    ) -> ChatMessage:
        """Record an assistant response."""
        msg = ChatMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            sources=sources or [],
            tools_used=tools_used or [],
        )
        self._messages.append(msg)
        return msg

    def add_tool_message(
        self,
        tool_call_id: str,
        name: str,
        content: str,
    ) -> ChatMessage:
        """Record a tool execution result."""
        msg = ChatMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
        self._messages.append(msg)
        return msg

    def get_messages_for_llm(
        self,
        max_messages: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get the list of messages formatted for the LLM chat completion API.
        Omits internal system messages since system prompt is passed separately.
        """
        messages = [
            msg.to_llm_dict()
            for msg in self._messages
            if msg.role in ("user", "assistant", "tool")
        ]

        if max_messages and len(messages) > max_messages:
            return messages[-max_messages:]

        return messages

    def get_display_messages(self) -> list[dict[str, Any]]:
        """
        Return user-facing messages (user and assistant with final content)
        for UI display.
        """
        return [
            msg.to_display_dict()
            for msg in self._messages
            if msg.role in ("user", "assistant") and msg.content
        ]

    def clear(self) -> None:
        """Clear all messages in this session."""
        self._messages.clear()
