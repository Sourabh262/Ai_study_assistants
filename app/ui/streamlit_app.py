import sys
import tempfile
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.agent.agent import StudyAssistantAgent
from app.exceptions import AIStudyAssistantError
from app.memory.conversation_memory import ConversationMemory
from app.rag.generator import LLMGenerator
from app.services.document_service import DocumentService

st.set_page_config(
    page_title="AI Study Assistant (Agentic RAG)",
    page_icon="🎓",
    layout="wide",
)


def initialize_session_state() -> None:
    """Initialize isolated per-session services and conversation memory."""
    if "document_service" not in st.session_state:
        st.session_state.document_service = DocumentService()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "llm" not in st.session_state:
        st.session_state.llm = LLMGenerator()

    if "agent" not in st.session_state:
        st.session_state.agent = StudyAssistantAgent(
            llm=st.session_state.llm,
            retriever=st.session_state.document_service.retriever,
        )


def process_uploaded_file(uploaded_file) -> None:
    """Process an uploaded study document and update the agent retriever."""
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        with st.spinner("Processing and indexing document chunks..."):
            chunk_count = st.session_state.document_service.process_file(
                temp_path,
                original_file_name=uploaded_file.name,
            )
            # Update the agent's active retriever
            st.session_state.agent.update_retriever(
                st.session_state.document_service.retriever
            )

        temp_path.unlink(missing_ok=True)

        st.success(
            f"Document '{uploaded_file.name}' processed successfully. "
            f"Indexed {chunk_count} chunks with page metadata."
        )

    except AIStudyAssistantError as exc:
        st.error(str(exc))

    except Exception:
        st.error("Failed to process the uploaded file. Please ensure it is a readable PDF or TXT.")


def render_sidebar() -> None:
    """Render the sidebar controls and document management."""
    with st.sidebar:
        st.title("🎓 Study Assistant")
        st.caption("Production-grade Agentic RAG System")

        st.write(
            "Upload study documents (PDF or TXT) and ask questions. "
            "The assistant autonomously decides whether to answer directly, "
            "retrieve from the document, or use the safe calculator."
        )

        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["txt", "pdf"],
            help="Upload a syllabus, lecture notes, textbook, or study material",
        )

        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True, type="primary"):
                process_uploaded_file(uploaded_file)

        st.divider()

        document_service = st.session_state.document_service

        if document_service.is_ready:
            st.success("📄 Document Ready")
            st.write(f"**File:** {document_service.file_name}")
            st.write(f"**Chunks:** {document_service.chunk_count}")

            if st.button("Clear Document", use_container_width=True):
                document_service.clear()
                st.session_state.agent.update_retriever(None)
                st.info("Document cleared.")
                st.rerun()
        else:
            st.info("ℹ️ No document loaded. Direct answering & math tools remain available.")

        st.divider()

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.memory.clear()
            st.success("Conversation cleared for this session.")
            st.rerun()


def render_chat() -> None:
    """Render chat messages and accept user input."""
    st.title("AI Study Assistant")
    st.caption("Ask questions, explore documents, or solve math problems. Conversation context is preserved across turns.")

    display_messages = st.session_state.memory.get_display_messages()

    for msg in display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander("📚 Retrieved Sources & Pages", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src}`")

            if msg.get("tools_used"):
                tools_str = ", ".join(msg["tools_used"])
                st.caption(f"🔧 Tools used: **{tools_str}**")

    question = st.chat_input("Ask a question, follow-up, or calculation...")

    if not question:
        return

    # Render user message
    with st.chat_message("user"):
        st.markdown(question)

    # Render assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and formulating answer..."):
            try:
                response = st.session_state.agent.run(
                    question=question,
                    memory=st.session_state.memory,
                )

                st.markdown(response.answer)

                if response.sources:
                    with st.expander("📚 Retrieved Sources & Pages", expanded=False):
                        for src in response.sources:
                            st.markdown(f"- `{src}`")

                # Show metadata badge
                tools_desc = (
                    ", ".join(response.tools_used)
                    if response.tools_used
                    else "direct_answer"
                )
                st.caption(
                    f"🔧 Tools: **{tools_desc}** | "
                    f"⏱️ Latency: **{response.latency_ms:.1f} ms** | "
                    f"🆔 Request: `{response.request_id}`"
                )

            except AIStudyAssistantError as exc:
                st.error(str(exc))
                st.session_state.memory.add_user_message(question)
                st.session_state.memory.add_assistant_message(f"Error: {str(exc)}")

            except Exception:
                error_msg = "An unexpected error occurred while processing your request. Please try again."
                st.error(error_msg)
                st.session_state.memory.add_user_message(question)
                st.session_state.memory.add_assistant_message(error_msg)


def main() -> None:
    initialize_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()