import json
import unittest
from types import SimpleNamespace
from typing import Any

from app.agent.agent import StudyAssistantAgent
from app.exceptions import AgentError
from app.memory.conversation_memory import ConversationMemory
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.vector_store import VectorStore
from app.tools.document_search import DocumentSearchTool
from app.tools.registry import create_default_registry


class MockChatCompletionMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = "stop" if not tool_calls else "tool_calls"


class MockLLMGenerator:
    """Mock LLM Generator that supports scripted or dynamic tool-calling behaviors."""

    def __init__(self, responses: list | None = None):
        self.responses = list(responses or [])
        self.call_history: list[dict[str, Any]] = []

    def chat_completion(self, messages: list[dict[str, Any]], tools: list | None = None, **kwargs):
        self.call_history.append({"messages": messages, "tools": tools})

        if self.responses:
            return self.responses.pop(0)

        # Fallback default: echo back direct answer
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return MockChatCompletionMessage(content=f"Direct response to: {last_user}")


def make_tool_call(call_id: str, name: str, args: dict[str, Any]):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class TestAgenticRAGSystem(unittest.TestCase):
    """
    Comprehensive test suite for the LLM-driven Agentic RAG system covering:
    - Direct questions
    - Calculator queries
    - Document questions
    - Follow-up/contextual questions
    - Multi-step/tool queries
    - Insufficient document information
    - Page refresh/session reset
    - Separate-user/session isolation
    - Hallucination prevention
    """

    def setUp(self):
        # Set up an in-memory vector store with sample study material
        from app.rag.embeddings import generate_embeddings
        self.store = VectorStore()
        texts = [
            "Data Structures CS101: Midterm exam will be held on October 15th on Chapters 1 to 4.",
            "Grading Policy: Homework is 30%, Midterm is 30%, Final Project is 40%.",
        ]
        metadatas = [
            {"source": "syllabus.pdf", "page": 1},
            {"source": "syllabus.pdf", "page": 2},
        ]
        embeddings = generate_embeddings(texts)
        self.store.add_documents(texts, embeddings, metadatas)
        self.retriever = Retriever(vector_store=self.store, top_k=2)
        self.doc_tool = DocumentSearchTool(self.retriever)

    def test_direct_question(self):
        """1. Direct question: LLM answers without invoking any tool."""
        mock_msg = MockChatCompletionMessage(
            content="Hello! I am your AI Study Assistant. How can I help you today?",
            tool_calls=None,
        )
        llm = MockLLMGenerator(responses=[mock_msg])
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        memory = ConversationMemory()

        response = agent.run("Hello!", memory=memory)

        self.assertIn("AI Study Assistant", response.answer)
        self.assertEqual(len(response.tools_used), 0)
        self.assertEqual(response.tool_used, "direct_llm")
        self.assertEqual(memory.message_count, 2)

    def test_calculator_query(self):
        """2. Calculator query: LLM emits tool call, application executes safe math, LLM returns final answer."""
        t_call = make_tool_call("call_calc", "calculate", {"expression": "25 * 4"})
        llm = MockLLMGenerator(
            responses=[
                MockChatCompletionMessage(tool_calls=[t_call]),
                MockChatCompletionMessage(content="The result of 25 * 4 is 100."),
            ]
        )
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        memory = ConversationMemory()

        response = agent.run("Calculate 25 * 4", memory=memory)

        self.assertIn("100", response.answer)
        self.assertIn("calculate", response.tools_used)
        self.assertEqual(memory.message_count, 2)

    def test_document_question(self):
        """3. Document question: LLM calls document_search, metadata is preserved, LLM grounds answer."""
        t_call = make_tool_call("call_doc", "document_search", {"query": "midterm exam date"})
        llm = MockLLMGenerator(
            responses=[
                MockChatCompletionMessage(tool_calls=[t_call]),
                MockChatCompletionMessage(
                    content="According to [syllabus.pdf, Page 1], the midterm exam will be held on October 15th."
                ),
            ]
        )
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        memory = ConversationMemory()

        response = agent.run("When is the midterm exam?", memory=memory)

        self.assertIn("October 15th", response.answer)
        self.assertIn("document_search", response.tools_used)
        self.assertTrue(any("syllabus.pdf" in src for src in response.sources))
        self.assertTrue(any("Page 1" in src for src in response.sources))

    def test_followup_and_contextual_questions(self):
        """4. Follow-up / contextual questions: Conversation history is passed to LLM."""
        llm = MockLLMGenerator(
            responses=[
                # Turn 1
                MockChatCompletionMessage(content="A Binary Search Tree is a node-based binary tree data structure."),
                # Turn 2
                MockChatCompletionMessage(content="It balances operations by ensuring left < root < right."),
            ]
        )
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        memory = ConversationMemory()

        # Turn 1
        agent.run("What is a BST?", memory=memory)

        # Turn 2: Follow-up relying on context
        agent.run("How does that work?", memory=memory)

        # Verify that turn 2 LLM call received turn 1 history
        last_call_messages = llm.call_history[-1]["messages"]
        user_msgs = [m["content"] for m in last_call_messages if m["role"] == "user"]
        self.assertIn("What is a BST?", user_msgs)
        self.assertIn("How does that work?", user_msgs)

    def test_multi_tool_query(self):
        """5. Multi-tool query: LLM invokes multiple tools (e.g. calculator + document_search)."""
        calc_call = make_tool_call("c1", "calculate", {"expression": "30 + 30 + 40"})
        doc_call = make_tool_call("d1", "document_search", {"query": "grading policy"})

        llm = MockLLMGenerator(
            responses=[
                MockChatCompletionMessage(tool_calls=[calc_call, doc_call]),
                MockChatCompletionMessage(
                    content="The total grade weight is 100%. According to syllabus.pdf, homework is 30%, midterm 30%, and final project 40%."
                ),
            ]
        )
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        memory = ConversationMemory()

        response = agent.run("Verify the grading percentages add to 100 and summarize them.", memory=memory)

        self.assertIn("calculate", response.tools_used)
        self.assertIn("document_search", response.tools_used)
        self.assertEqual(response.tool_used, "multi_tool")
        self.assertIn("100%", response.answer)

    def test_insufficient_document_information_and_hallucination_prevention(self):
        """6 & 9. Insufficient info & hallucination prevention: LLM states information not found instead of fabricating."""
        doc_call = make_tool_call("d1", "document_search", {"query": "professor office phone number"})
        llm = MockLLMGenerator(
            responses=[
                MockChatCompletionMessage(tool_calls=[doc_call]),
                MockChatCompletionMessage(
                    content="The uploaded document does not contain information regarding the professor's phone number."
                ),
            ]
        )
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        memory = ConversationMemory()

        response = agent.run("What is the professor's phone number?", memory=memory)

        self.assertIn("does not contain", response.answer.lower())
        self.assertNotIn("555", response.answer)

    def test_separate_user_and_session_isolation(self):
        """7 & 8. Session isolation & Page refresh reset: Two distinct sessions never mix."""
        user1_memory = ConversationMemory(session_id="session_user1")
        user2_memory = ConversationMemory(session_id="session_user2")

        llm = MockLLMGenerator(
            responses=[
                MockChatCompletionMessage(content="User 1 response"),
                MockChatCompletionMessage(content="User 2 response"),
            ]
        )
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)

        agent.run("User 1 question about Physics", memory=user1_memory)
        agent.run("User 2 question about Chemistry", memory=user2_memory)

        u1_history = [m["content"] for m in user1_memory.get_messages_for_llm()]
        u2_history = [m["content"] for m in user2_memory.get_messages_for_llm()]

        self.assertTrue(any("Physics" in h for h in u1_history))
        self.assertFalse(any("Chemistry" in h for h in u1_history))

        self.assertTrue(any("Chemistry" in h for h in u2_history))
        self.assertFalse(any("Physics" in h for h in u2_history))

        # Page refresh simulation
        user1_memory.clear()
        self.assertEqual(user1_memory.message_count, 0)
        self.assertEqual(len(user1_memory.get_messages_for_llm()), 0)

    def test_empty_question_validation(self):
        """Ensure empty or whitespace queries are rejected."""
        llm = MockLLMGenerator()
        agent = StudyAssistantAgent(llm=llm, retriever=self.retriever)
        with self.assertRaises(AgentError):
            agent.run("")
        with self.assertRaises(AgentError):
            agent.run("   ")


if __name__ == "__main__":
    unittest.main()
