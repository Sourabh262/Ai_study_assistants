import unittest

from app.rag.chunker import split_pages, split_text


class TestChunker(unittest.TestCase):
    def test_split_text(self):
        sample_text = "This is a paragraph. " * 50
        chunks = split_text(sample_text)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)

    def test_split_pages_metadata(self):
        pages = [
            ("First page content of the study guide.", 1),
            ("Second page content with detailed explanations.", 2),
        ]
        chunk_data = split_pages(pages, source_name="guide.pdf")
        self.assertGreater(len(chunk_data), 0)

        chunk_text, meta = chunk_data[0]
        self.assertIn("guide.pdf", meta["source"])
        self.assertEqual(meta["page"], 1)


if __name__ == "__main__":
    unittest.main()
