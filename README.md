# AI Study Assistant (Production-Grade Agentic RAG)

> An intelligent, production-level AI Study Assistant built with LLM-driven Agentic Retrieval-Augmented Generation (RAG), dynamic function calling, safe AST computation, and session-isolated conversation memory.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Inference-Groq%20Cloud-F55036?style=flat)](https://groq.com/)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow?style=flat)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(29%2F29)-brightgreen?style=flat)]()

---

## 1. Architecture & Design Principles

The assistant operates as a **production-grade Agentic RAG system** driven by the LLM:

* **Core Principle:** The **LLM understands the user's intent and decides whether to answer directly or invoke registered tools**, while the **application controls tool availability, schema validation, sandboxed execution, security, and session memory**.
* **Zero Hardcoded Routing:** Absolutely no `if/elif`, regex patterns, or fixed intent lists determine tool choice. Routing is dynamic and model-driven through standard function calling schemas.
* **Autonomous Multi-Tool Orchestration:** Supports calling one or multiple tools (e.g. arithmetic + document search) in parallel or sequentially within a ReAct-style loop.
* **Strict Grounding & Hallucination Prevention:** Document queries are answered strictly based on retrieved chunks. When information is not present, the assistant explicitly states so rather than fabricating facts.
* **Complete Conversation Memory:** Full multi-turn context is maintained per session so follow-ups (`"why?"`, `"explain that in detail"`, `"calculate again"`) are seamlessly understood.

---

## 2. Agent + RAG Flow

```text
User Query + Conversation History
        ↓
    LLM Agent
        ↓
Direct Answer OR Tool Call(s)
        ↓
Safe Application Execution (ToolRegistry)
        ↓
Tool Result(s) with Metadata
        ↓
       LLM
        ↓
Final Grounded Answer + Source/Page Citations
        ↓
Update Session History
```

---

## 3. Registered Tools

| Tool | Schema Name | Description | Security & Safety |
| :--- | :--- | :--- | :--- |
| **Calculator** | `calculate` | Safely evaluates mathematical expressions (`+`, `-`, `*`, `/`, `%`, `**`, parentheses, percentages). | Evaluated using Python AST node traversal. **Never** uses `eval()`, strictly blocks imports, functions, and arbitrary code. |
| **Document Search** | `document_search` | Dense vector semantic search over uploaded PDF/TXT study materials. | Returns relevant chunks with cosine similarity scores, file name, and page number metadata. |

---

## 4. Conversation Memory & Isolation

- **In-Memory & Session Isolated:** Implemented in `app/memory/conversation_memory.py`. Each browser tab/user session maintains an isolated `ConversationMemory` instance in `st.session_state`.
- **Zero Database Persistence for Chat:** A full page refresh cleanly starts a new conversation session without database residue.
- **Contextual Query Reformulation:** When the LLM decides to search the document on a follow-up turn, it reformulates the search query into a standalone topic query using the conversation history.

---

## 5. Project Directory Structure

```text
ai-study-assistant/
├── app/
│   ├── agent/
│   │   ├── agent.py               # Production StudyAssistantAgent & tool loop
│   │   └── router.py              # Tool identifiers & backward compatibility
│   ├── config.py                  # Environment and model configurations
│   ├── exceptions.py              # Custom typed exception hierarchy
│   ├── loaders/
│   │   ├── pdf_loader.py          # PDF loader extracting page-level text
│   │   └── text_loader.py         # TXT loader
│   ├── logging_config.py          # Structured logging (Request IDs, latency)
│   ├── memory/
│   │   └── conversation_memory.py # Session-isolated multi-turn memory
│   ├── rag/
│   │   ├── chunker.py             # Recursive splitter preserving page metadata
│   │   ├── embeddings.py          # SentenceTransformers all-MiniLM-L6-v2
│   │   ├── generator.py           # Groq client wrapper with native tool calling
│   │   ├── prompts.py             # System prompt with strict grounding rules
│   │   ├── retriever.py           # Semantic retriever with metadata formatting
│   │   └── vector_store.py        # In-memory NumPy cosine similarity store
│   ├── services/
│   │   └── document_service.py    # Document ingestion pipeline
│   ├── tools/
│   │   ├── calculator.py          # Safe AST-based calculator (no eval)
│   │   ├── document_search.py     # Document retrieval tool wrapper
│   │   └── registry.py            # Central ToolRegistry with schema validation
│   └── ui/
│       └── streamlit_app.py       # Production Streamlit UI
├── tests/
│   ├── test_agent.py              # 9 comprehensive Agentic RAG scenarios
│   ├── test_calculator.py         # Calculator AST tests and security rejection
│   ├── test_chunker.py            # Chunking and metadata tests
│   ├── test_document_loader.py    # Document loading tests
│   ├── test_memory.py             # Conversation memory & session isolation tests
│   └── test_retriever.py          # Vector store & retrieval tests
├── requirements.txt
└── README.md
```

---

## 6. Running the Application

### 1. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
CHUNK_SIZE=700
CHUNK_OVERLAP=120
TOP_K=5
MIN_SIMILARITY=0.05
```

### 2. Run Streamlit UI
```bash
streamlit run app/ui/streamlit_app.py
```

### 3. Run Automated Test Suite
```bash
pytest tests/
```
All 29 tests validate direct questions, safe math calculations, document retrieval, multi-tool queries, contextual follow-ups, hallucination prevention, and session isolation.
