import unittest
from africa_tts.chunking import chunk_paragraphs


class TestChunking(unittest.TestCase):
    def test_single_sentence(self):
        text = "Obiara wɔ nkwa, fawohodie ne ne bammɔ ho kyɛfa."
        chunks = chunk_paragraphs(text)
        self.assertEqual(chunks, [text])

    def test_multiple_paragraphs(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = chunk_paragraphs(text)
        self.assertEqual(chunks, ["First paragraph here.", "Second paragraph here."])

    def test_long_run_on_sentence_clause_splitting(self):
        words = ["word"] * 40
        text = " ".join(words)
        chunks = chunk_paragraphs(text, max_words_per_clause=20)
        self.assertEqual(len(chunks), 1)
        self.assertIn(",", chunks[0])
        self.assertTrue(chunks[0].endswith("."))


if __name__ == "__main__":
    unittest.main()
