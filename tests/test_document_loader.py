import tempfile
import unittest
from pathlib import Path

from app.exceptions import EmptyDocumentError
from app.loaders.text_loader import load_text_file


class TestDocumentLoader(unittest.TestCase):
    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write("Artificial intelligence is transforming education.")
            temp_path = Path(tf.name)

        try:
            content = load_text_file(temp_path)
            self.assertIn("Artificial intelligence", content)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_empty_text_file(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write("   \n\n  ")
            temp_path = Path(tf.name)

        try:
            with self.assertRaises(EmptyDocumentError):
                load_text_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
