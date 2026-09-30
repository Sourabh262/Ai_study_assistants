import re
from enum import Enum

from app.logging_config import logger


class ToolName(str, Enum):
    CALCULATOR = "calculator"
    DOCUMENT_SEARCH = "document_search"
    DIRECT_LLM = "direct_llm"
    MULTI_TOOL = "multi_tool"


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
    "book",
    "summary",
    "summarize",
    "conclusion",
    "overview",
    "uploaded",
    "my name",
    "my cgpa",
    "my gpa",
    "my skills",
    "my experience",
    "my projects",
    "who am i",
}


def is_document_query(question: str) -> bool:
    """Check if the user is explicitly inquiring about an uploaded document."""
    if not question:
        return False
    normalized = question.strip().lower()
    if any(keyword in normalized for keyword in DOCUMENT_KEYWORDS):
        return True
    if re.search(r"\b\w+\.(txt|pdf)\b", normalized):
        return True
    return False


GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|greetings|howdy|good morning|good afternoon|good evening)[!.,? ]*$",
    re.IGNORECASE,
)


GENERAL_QUERY_PATTERN = re.compile(
    r"""
    (
        \b(today['\s]*s?\s*date|current\s*date|what\s+is\s+the\s+(today\s+)?date|date\s+is\s+what|what\s+date\s+is\s+it|what\s+day\s+is\s+it)\b
        |
        \b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|how\s+are\s+you|how\s+do\s+you\s+do|who\s+(made|created)\s+you)\b
        |
        \b(tell\s+me\s+a\s+joke|say\s+a\s+joke|make\s+me\s+laugh|write\s+a\s+poem|tell\s+a\s+story)\b
        |
        ^(thanks|thank\s+you|thanks\s+a\s+lot|bye|goodbye|see\s+you|ok|okay|cool|nice|great)[!.,? ]*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_compound_math_question(question: str) -> bool:
    """
    Check if a question containing math also asks an additional non-math question.
    """
    cleaned = question.lower()
    # Remove numbers and math symbols
    cleaned = re.sub(r"[\d\+\-\*\/\%\^\(\)\=]", " ", cleaned)
    # Remove common math query filler words
    math_fillers = [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\bevaluate\b",
        r"\bwhat is\b",
        r"\bwhat's\b",
        r"\bhow much is\b",
        r"\bresult of\b",
        r"\bvalue of\b",
        r"\bthe answer\b",
        r"\bof\b",
        r"\band\b",
        r"\bplus\b",
        r"\bminus\b",
        r"\btimes\b",
        r"\bdivided by\b",
        r"\bmultiplied by\b",
        r"\bequals\b",
        r"\bplease\b",
    ]
    for filler in math_fillers:
        cleaned = re.sub(filler, " ", cleaned)

    words = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", cleaned)]
    return len(words) > 0


def route_question(question: str, has_document: bool = False) -> ToolName:
    """
    Decide which tool should handle the user's question.

    Priority:
    1. Pure greetings or general date/identity questions -> Direct LLM
    2. Compound questions (concept + math) -> Multi Tool
    3. Pure mathematical questions -> Calculator
    4. Explicit document keywords -> Document Search
    5. If document is loaded -> Document Search
    6. Fallback -> Direct LLM
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
        if is_compound_math_question(normalized):
            logger.info(
                "Agent route: MULTI_TOOL (compound question with math) | question=%s",
                question,
            )
            return ToolName.MULTI_TOOL

        logger.info(
            "Agent route: CALCULATOR | question=%s",
            question,
        )
        return ToolName.CALCULATOR

    # Explicit document keywords
    if is_document_query(normalized):
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