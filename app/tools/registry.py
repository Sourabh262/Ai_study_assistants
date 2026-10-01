import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from app.logging_config import logger


@dataclass
class ToolDefinition:
    """Registered tool with name, description, parameters schema, and handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""

    name: str
    output: str
    success: bool
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry for application tools.

    Ensures:
    - Clear schemas/descriptions for LLM function calling
    - Strict validation of tool arguments
    - Safe sandboxed execution (LLM cannot execute arbitrary code)
    - Structured logging of all tool invocations
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.info("Tool registered: %s", tool.name)

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """
        Generate OpenAI/Groq compatible tools specification.
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return schemas

    def execute(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        """
        Safely execute a registered tool with argument validation.

        Args:
            name: Tool name.
            arguments: Dictionary or JSON string of arguments.

        Returns:
            ToolResult containing execution output or safe error message.
        """
        start_time = time.perf_counter()

        tool = self.get_tool(name)
        if not tool:
            logger.error("Attempted to execute unregistered tool: %s", name)
            return ToolResult(
                name=name,
                output=f"Error: Tool '{name}' is not registered or supported.",
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Parse string arguments if received as JSON string
        parsed_args: dict[str, Any]
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except Exception as exc:
                logger.error("Failed to parse tool arguments JSON for '%s': %s", name, exc)
                return ToolResult(
                    name=name,
                    output=f"Error: Invalid arguments JSON: {exc}",
                    success=False,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments
        else:
            return ToolResult(
                name=name,
                output="Error: Arguments must be a JSON object.",
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Validate required arguments against schema
        required_params = tool.parameters.get("required", [])
        for req in required_params:
            if req not in parsed_args:
                msg = f"Missing required parameter '{req}' for tool '{name}'."
                logger.warning(msg)
                return ToolResult(
                    name=name,
                    output=f"Error: {msg}",
                    success=False,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Execute handler safely
        try:
            logger.info("Executing tool: %s with args: %s", name, parsed_args)
            raw_output = tool.handler(**parsed_args)

            latency = (time.perf_counter() - start_time) * 1000
            output_str = str(raw_output)
            logger.info("Tool %s executed successfully in %.2f ms", name, latency)

            return ToolResult(
                name=name,
                output=output_str,
                success=True,
                latency_ms=latency,
            )

        except ToolError as exc:
            latency = (time.perf_counter() - start_time) * 1000
            logger.warning("Tool %s raised business exception: %s", name, exc)
            return ToolResult(
                name=name,
                output=f"Tool error: {exc}",
                success=False,
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error("Tool %s failed unexpectedly: %s", name, exc)
            return ToolResult(
                name=name,
                output=f"Tool execution failed: {exc}",
                success=False,
                latency_ms=latency,
            )


def create_default_registry(document_search_tool: Any = None) -> ToolRegistry:
    """
    Factory creating a default ToolRegistry with Calculator and Document Search tools.
    """
    from app.tools.calculator import execute_calculate

    registry = ToolRegistry()

    # Register Calculator tool
    calc_def = ToolDefinition(
        name="calculate",
        description=(
            "Safely evaluates mathematical expressions (arithmetic +, -, *, /, %, **, "
            "parentheses, percentages like '15% of 200'). "
            "Always use this tool for any mathematical calculations, arithmetic questions, or numeric computations."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '25 * 4', '15% of 200', '((12 + 8) * 3) / 2'",
                }
            },
            "required": ["expression"],
        },
        handler=execute_calculate,
    )
    registry.register(calc_def)

    # Register Document Search tool
    if document_search_tool is not None:
        doc_def = ToolDefinition(
            name="document_search",
            description=(
                "Searches the uploaded study document (PDF or TXT) for semantically relevant passages "
                "using vector retrieval. "
                "Use this tool whenever the user asks questions about uploaded documents, study notes, "
                "resume, syllabus, or document-specific facts. "
                "Returns relevant chunks with source file and page numbers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query or concept to look up in the uploaded document. "
                            "When answering follow-up questions, formulate a standalone, descriptive search query "
                            "incorporating necessary context from previous turns."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of relevant chunks to retrieve (default is 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=document_search_tool.execute,
        )
        registry.register(doc_def)

    return registry
