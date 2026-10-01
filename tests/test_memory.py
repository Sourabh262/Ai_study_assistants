import unittest

from app.memory.conversation_memory import ChatMessage, ConversationMemory


class TestConversationMemory(unittest.TestCase):
    """Tests for conversation memory, session isolation, and serialization."""

    def test_message_addition_and_counts(self):
        memory = ConversationMemory()
        self.assertEqual(memory.message_count, 0)

        memory.add_user_message("Hello, assistant!")
        memory.add_assistant_message("Hello! How can I help you study today?")

        self.assertEqual(memory.message_count, 2)
        display_msgs = memory.get_display_messages()
        self.assertEqual(len(display_msgs), 2)
        self.assertEqual(display_msgs[0]["role"], "user")
        self.assertEqual(display_msgs[0]["content"], "Hello, assistant!")
        self.assertEqual(display_msgs[1]["role"], "assistant")

    def test_llm_messages_formatting(self):
        memory = ConversationMemory()
        memory.add_user_message("What is 10 + 5?")
        memory.add_tool_message(
            tool_call_id="call_123",
            name="calculate",
            content="15",
        )
        memory.add_assistant_message(
            content="The answer is 15.",
            tools_used=["calculate"],
        )

        llm_msgs = memory.get_messages_for_llm()
        self.assertEqual(len(llm_msgs), 3)
        self.assertEqual(llm_msgs[0]["role"], "user")
        self.assertEqual(llm_msgs[1]["role"], "tool")
        self.assertEqual(llm_msgs[1]["tool_call_id"], "call_123")
        self.assertEqual(llm_msgs[2]["role"], "assistant")

    def test_session_isolation(self):
        """Verify that two distinct user/session memories never cross-contaminate."""
        session_a = ConversationMemory(session_id="user_alice")
        session_b = ConversationMemory(session_id="user_bob")

        session_a.add_user_message("Alice's confidential study notes")
        session_b.add_user_message("Bob's physics question")

        self.assertEqual(session_a.message_count, 1)
        self.assertEqual(session_b.message_count, 1)

        self.assertIn("Alice", session_a.get_messages_for_llm()[0]["content"])
        self.assertNotIn("Alice", session_b.get_messages_for_llm()[0]["content"])
        self.assertIn("Bob", session_b.get_messages_for_llm()[0]["content"])
        self.assertNotIn("Bob", session_a.get_messages_for_llm()[0]["content"])

    def test_page_refresh_session_reset(self):
        """Simulate page refresh or manual session reset."""
        memory = ConversationMemory()
        memory.add_user_message("Query 1")
        memory.add_assistant_message("Answer 1")
        self.assertEqual(memory.message_count, 2)

        # Full reset
        memory.clear()
        self.assertEqual(memory.message_count, 0)
        self.assertEqual(len(memory.get_messages_for_llm()), 0)
        self.assertEqual(len(memory.get_display_messages()), 0)


if __name__ == "__main__":
    unittest.main()
