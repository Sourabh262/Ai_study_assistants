import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile

import streamlit as st

from app.agent.agent import StudyAssistantAgent
from app.exceptions import AIStudyAssistantError
from app.rag.generator import LLMGenerator
from app.services.document_service import DocumentService


st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="AI",
    layout="wide",
)


def initialize_session_state() -> None:

    if "document_service" not in st.session_state:
        st.session_state.document_service = DocumentService()

    if "llm" not in st.session_state:
        st.session_state.llm = LLMGenerator()

    if "agent" not in st.session_state:
        st.session_state.agent = StudyAssistantAgent(
            llm=st.session_state.llm,
            retriever=st.session_state.document_service.retriever,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []


def process_uploaded_file(uploaded_file) -> None:

    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        with st.spinner("Processing document..."):

            chunk_count = (
                st.session_state.document_service.process_file(
                    temp_path,
                    original_file_name=uploaded_file.name,
                )
            )

        temp_path.unlink(missing_ok=True)

        st.session_state.messages = []

        st.success(
            f"Document processed successfully. "
            f"Created {chunk_count} chunks."
        )

    except AIStudyAssistantError as exc:

        st.error(str(exc))

    except Exception as exc:

        st.error(
            "Something went wrong while processing the document."
        )

        st.exception(exc)


def render_sidebar() -> None:

    with st.sidebar:

        st.title("AI Study Assistant")

        st.write(
            "Upload a TXT or PDF document and ask questions "
            "about it."
        )

        uploaded_file = st.file_uploader(
            "Upload document",
            type=["txt", "pdf"],
        )

        if uploaded_file is not None:

            if st.button(
                "Process Document",
                use_container_width=True,
            ):
                process_uploaded_file(uploaded_file)

        st.divider()

        document_service = st.session_state.document_service

        if document_service.is_ready:

            st.success("Document ready")

            st.write(
                f"**File:** {document_service.file_name}"
            )

            st.write(
                f"**Chunks:** {document_service.chunk_count}"
            )

            if st.button(
                "Clear Document",
                use_container_width=True,
            ):

                document_service.clear()
                st.session_state.messages = []

                st.rerun()

        else:

            st.info(
                "No document loaded."
            )


def render_chat() -> None:

    st.title("AI Study Assistant")

    st.caption(
        "Ask questions, search your document, "
        "or solve mathematical problems."
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("Sources"):

                    for source in message["sources"]:
                        st.write(source)

            if message.get("tool"):

                st.caption(
                    f"Tool used: {message['tool']}"
                )

    question = st.chat_input(
        "Ask something..."
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = (
                    st.session_state.agent.run(question)
                )

                st.markdown(response.answer)

                if response.sources:

                    with st.expander("Sources"):

                        for source in response.sources:
                            st.write(source)

                st.caption(
                    f"Tool used: {response.tool_used.value}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "tool": response.tool_used.value,
                    }
                )

            except AIStudyAssistantError as exc:

                error_message = str(exc)

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except Exception:

                error_message = (
                    "Something went wrong while "
                    "processing your question."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


def main() -> None:

    initialize_session_state()

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()