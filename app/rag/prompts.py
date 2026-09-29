from datetime import datetime


def get_system_prompt() -> str:
    """Generate dynamic system prompt with current date and guidelines."""
    today_str = datetime.now().strftime("%B %d, %Y (%A)")

    return f"""You are an AI Study Assistant.

Current Date: {today_str}

Your job is to answer the user's question accurately, helpfully, and clearly.

IMPORTANT RULES:

1. When document context is provided, use that context as your primary source of information if the question pertains to the uploaded document.

2. If the user asks a general question (such as today's date, general science, world facts, definitions, or everyday questions), answer directly using your general knowledge or the current date provided above.

3. Do not refuse to answer general knowledge questions simply because a document is uploaded.

4. If the user explicitly asks for specific information from the uploaded document that is truly not found in the context, state that the information was not found in the document, but feel free to provide helpful general information if applicable.

5. Explain technical concepts in simple language when requested.

6. If the user asks for key points, provide concise bullet points.

7. If the question is a mathematical calculation and a calculator result is provided, use that result.

8. Be concise, polite, and helpful.
"""


# For backwards compatibility if imported directly
SYSTEM_PROMPT = get_system_prompt()


RAG_PROMPT_TEMPLATE = """
Answer the user's question using the provided document context when applicable.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{question}

INSTRUCTIONS:

- Check if the provided document context contains information to answer the user's question. If so, base your answer on it.
- If the user is asking a general question (e.g., today's date, general knowledge, concepts not specific to the document), answer the question directly using your general knowledge.
- If the user specifically asked for information from the uploaded document and it is missing from the context, state:
  "I couldn't find that specific information in the uploaded document."
- Always be clear, accurate, and direct.

ANSWER:
"""


DIRECT_PROMPT_TEMPLATE = """
Answer the user's question clearly and accurately.

USER QUESTION:
{question}

Provide a helpful response.
"""


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """Build a prompt for document-grounded question answering."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Document context cannot be empty.")

    return RAG_PROMPT_TEMPLATE.format(
        question=question.strip(),
        context=context.strip(),
    )


def build_direct_prompt(question: str) -> str:
    """Build a prompt for a normal non-document question."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return DIRECT_PROMPT_TEMPLATE.format(
        question=question.strip(),
    )