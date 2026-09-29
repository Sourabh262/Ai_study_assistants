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
Answer the user's question clearly and accurately.

DOCUMENT CONTEXT FROM UPLOADED FILE:
-----------------
{context}
-----------------

USER QUESTION:
{question}

INSTRUCTIONS:

1. If the provided document context contains relevant details to answer the question (e.g., specific details about a project, person, grades, resume details, or document topics), prioritize and reference those document details.
2. If the user's question is a general concept, programming language definition, educational topic, or general knowledge question (such as "what is python", "explain machine learning", "what is photosynthesis", etc.), ALWAYS provide a complete, clear, and helpful explanation using your general knowledge, integrating any document context if relevant.
3. NEVER reply with refusal phrases like "I couldn't find that information in the uploaded document" for general knowledge or conceptual questions. Provide the answer directly to the user.

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