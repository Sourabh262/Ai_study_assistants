class AIStudyAssistantError(Exception):
    """Base exception for the AI Study Assistant."""

    pass


class ConfigurationError(AIStudyAssistantError):
    """Raised when application configuration is invalid."""

    pass


class DocumentError(AIStudyAssistantError):
    """Raised when document processing fails."""

    pass


class UnsupportedFileTypeError(DocumentError):
    """Raised when an unsupported document type is uploaded."""

    pass


class EmptyDocumentError(DocumentError):
    """Raised when a document contains no readable text."""

    pass


class DocumentSearchError(AIStudyAssistantError):
    """Raised when document search fails."""

    pass


class EmbeddingError(AIStudyAssistantError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(AIStudyAssistantError):
    """Raised when vector store operations fail."""

    pass


class CalculatorError(AIStudyAssistantError):
    """Raised when calculator execution fails."""

    pass


class AgentError(AIStudyAssistantError):
    """Raised when agent processing fails."""

    pass


class ToolError(AIStudyAssistantError):
    """Raised when tool registration, validation, or execution fails."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found in the registry."""

    pass


class ToolValidationError(ToolError):
    """Raised when tool input arguments fail schema validation."""

    pass


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""

    pass


class LLMError(AIStudyAssistantError):
    """Raised when the LLM request fails."""

    pass


class RateLimitError(LLMError):
    """Raised when the LLM provider returns a rate limit response."""

    pass