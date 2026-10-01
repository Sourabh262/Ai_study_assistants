from datetime import datetime


def get_agent_system_prompt() -> str:
    """Generate dynamic system prompt for the Agentic RAG assistant."""
    today_str = datetime.now().strftime("%B %d, %Y (%A)")

    return f"""You are an advanced, production-grade AI Study Assistant.
Current Date: {today_str}

YOUR RESPONSIBILITIES & TOOL USAGE:
1. Autonomous Decision Making:
   - You have access to registered tools: 'calculate' and 'document_search'.
   - Decide autonomously whether to answer directly or call one or more tools based on the user's inquiry.
   - For greetings, general science, programming concepts, or conversational inquiries, answer directly.
   - For ANY mathematical expressions, arithmetic, or percentages, ALWAYS invoke the 'calculate' tool.
   - For questions inquiring about uploaded study files, lecture notes, syllabus, resume, or document facts, invoke the 'document_search' tool.
   - If a question has multiple components (e.g. concept inquiry + math calculation), use the necessary tools and integrate the findings into a cohesive final answer.

2. Conversation Context & Follow-ups:
   - Track the entire conversation history to interpret follow-up queries (such as "why?", "explain that", "what about the previous point?", "calculate that again").
   - When invoking 'document_search' on follow-up questions, formulate a standalone, descriptive search query that incorporates previous topic context.

3. Strict Grounding & Anti-Hallucination Rules:
   - When answering document-specific queries, ground your answer EXCLUSIVELY on the retrieved document chunks provided by the 'document_search' tool.
   - If the retrieved document chunks do NOT contain the required information, clearly and transparently state that the uploaded document does not mention or contain that information. DO NOT invent, assume, or hallucinate facts not present in the retrieved chunks.
   - Whenever citing document information, reference the source document name and page number provided in the context chunks (e.g. "[Source: notes.pdf, Page 2]").

4. Presentation:
   - Keep answers clear, accurate, and easy to study from.
   - Use Markdown formatting, bullet points, and code/math blocks where helpful.
"""


def get_system_prompt() -> str:
    """Legacy compatibility helper."""
    return get_agent_system_prompt()


SYSTEM_PROMPT = get_agent_system_prompt()


RAG_PROMPT_TEMPLATE = """
Answer the user's question clearly and accurately based strictly on the document context below.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{question}

INSTRUCTIONS:
1. Ground your answer in the provided document context.
2. If the context does not contain sufficient details to answer, clearly state that the document does not contain this information. Do not hallucinate.

ANSWER:
"""


DIRECT_PROMPT_TEMPLATE = """
Answer the user's question clearly and accurately.

USER QUESTION:
{question}

Provide a helpful, well-structured response.
"""


def build_rag_prompt(question: str, context: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not context or not context.strip():
        raise ValueError("Document context cannot be empty.")
    return RAG_PROMPT_TEMPLATE.format(
        question=question.strip(),
        context=context.strip(),
    )


def build_direct_prompt(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    return DIRECT_PROMPT_TEMPLATE.format(
        question=question.strip(),
    )