import unittest

from app.rag.retriever import Retriever, format_retrieved_context
from app.rag.vector_store import VectorStore


class TestRetrieverAndVectorStore(unittest.TestCase):
    """Tests for vector store, metadata preservation, and semantic retrieval."""

    def setUp(self):
        self.store = VectorStore()
        # Mock embeddings for testing without needing ML model download in unit test
        self.texts = [
            "Photosynthesis is the process by which green plants create food.",
            "Newton's laws of motion describe the relationship between a body and forces.",
            "A binary search tree is a rooted binary tree data structure.",
        ]
        self.metadatas = [
            {"source": "biology.pdf", "page": 12},
            {"source": "physics.txt", "page": 1},
            {"source": "cs.pdf", "page": 45},
        ]
        # 3-dim orthogonal-ish vectors for deterministic testing
        self.embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.store.add_documents(self.texts, self.embeddings, self.metadatas)
        self.retriever = Retriever(vector_store=self.store, top_k=2, min_similarity=0.1)

    def test_vector_store_is_ready(self):
        self.assertTrue(self.store.is_ready)
        self.assertEqual(self.store.document_count, 3)

    def test_search_with_metadata(self):
        # Querying close to photosynthesis (vector [0.9, 0.1, 0.0])
        results = self.store.search(query_embedding=[0.9, 0.1, 0.0], top_k=2)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top.text, self.texts[0])
        self.assertEqual(top.metadata.get("source"), "biology.pdf")
        self.assertEqual(top.metadata.get("page"), 12)

    def test_format_retrieved_context(self):
        results = self.store.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)
        chunks = [
            type("Chunk", (), {"text": r.text, "score": r.score, "metadata": r.metadata})()
            for r in results
        ]
        context = format_retrieved_context(chunks)
        self.assertIn("biology.pdf", context)
        self.assertIn("Page: 12", context)
        self.assertIn("Photosynthesis", context)

    def test_clear_vector_store(self):
        self.store.clear()
        self.assertFalse(self.store.is_ready)
        self.assertEqual(self.store.document_count, 0)


if __name__ == "__main__":
    unittest.main()
