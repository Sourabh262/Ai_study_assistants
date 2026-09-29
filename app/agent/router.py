import re
from enum import Enum

from app.logging_config import logger


class ToolName(str, Enum):
    CALCULATOR = "calculator"
    DOCUMENT_SEARCH = "document_search"
    DIRECT_LLM = "direct_llm"


MATH_PATTERN = re.compile(
    r"""
    (
        \d+\s*[\+\-\*\/\%\^]\s*\d+
        |
        \d+(?:\.\d+)?\s*%\s*of\s*\d+(?:\.\d+)?
        |
        \b(calculate|compute|solve|evaluate)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

DOCUMENT_KEYWORDS = {
    "according to the document",
    "according to the file",
    "in the document",
    "in the file",
    "from the document",
    "from the file",
    "uploaded document",
    "uploaded file",
    "this document",
    "this file",
    "pdf",
    "document",
    "resume",
    "cv",
    "notes",
    "syllabus",
    "paper",
    "assignment",
    "report",
    "my name",
    "my cgpa",
    "my gpa",
    "my skills",
    "my experience",
    "my projects",
    "who am i",
}

GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|greetings|howdy|good morning|good afternoon|good evening)[!.,? ]*$",
    re.IGNORECASE,
)

GENERAL_QUERY_PATTERN = re.compile(
    r"""
    (
        \b(today['\s]*s?\s*date|current\s*date|what\s+is\s+the\s+(today\s+)?date|date\s+is\s+what|what\s+date\s+is\s+it|what\s+day\s+is\s+it)\b
        |
        \b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def route_question(question: str, has_document: bool = False) -> ToolName:
    """
    Decide which tool should handle the user's question.

    Priority:
    1. Pure greetings or general date/identity questions -> Direct LLM
    2. Mathematical questions -> Calculator
    3. Explicit document keywords -> Document Search
    4. If document is loaded -> Document Search
    5. Fallback -> Direct LLM
    """

    if not question or not question.strip():
        return ToolName.DIRECT_LLM

    normalized = question.strip().lower()

    # Pure greetings stay direct LLM
    if GREETING_PATTERN.match(normalized):
        logger.info(
            "Agent route: DIRECT_LLM (greeting) | question=%s",
            question,
        )
        return ToolName.DIRECT_LLM

    # General queries like today's date or bot identity stay direct LLM
    if GENERAL_QUERY_PATTERN.search(normalized):
        logger.info(
            "Agent route: DIRECT_LLM (general query) | question=%s",
            question,
        )
        return ToolName.DIRECT_LLM

    # Mathematical expressions/questions.
    if MATH_PATTERN.search(normalized):
        logger.info(
            "Agent route: CALCULATOR | question=%s",
            question,
        )
        return ToolName.CALCULATOR

    # Explicit document keywords
    if any(keyword in normalized for keyword in DOCUMENT_KEYWORDS):
        logger.info(
            "Agent route: DOCUMENT_SEARCH (keyword match) | question=%s",
            question,
        )
        return ToolName.DOCUMENT_SEARCH

    # If a document is uploaded and ready, prioritize document search
    if has_document:
        logger.info(
            "Agent route: DOCUMENT_SEARCH (document is active) | question=%s",
            question,
        )
        return ToolName.DOCUMENT_SEARCH

    logger.info(
        "Agent route: DIRECT_LLM | question=%s",
        question,
    )

    return ToolName.DIRECT_LLM