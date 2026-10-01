from enum import Enum


class ToolName(str, Enum):
    """
    Tool identifier enum preserved for backwards-compatibility.
    Note: Dynamic routing is performed by the LLM agent via tool calling schemas.
    """

    CALCULATOR = "calculate"
    DOCUMENT_SEARCH = "document_search"
    DIRECT_LLM = "direct_llm"
    MULTI_TOOL = "multi_tool"