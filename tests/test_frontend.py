import unittest
from africa_tts.frontend import to_ipa


class TestFrontend(unittest.TestCase):
    def test_african_language_pure(self):
        ipa = to_ipa("Akwaaba!", "twi")
        self.assertIn("akʷaaba", ipa)

    def test_codeswitching_single_word(self):
        ipa = to_ipa("Mepɛ sɛ mekɔ school.", "twi", codeswitch=True)
        self.assertIn("skuːl", ipa)
        self.assertIn("mepɛ", ipa)

    def test_codeswitching_multi_word_phrase(self):
        ipa = to_ipa("Mepɛ sɛ mekɔ high school wɔ regional hospital no nkyɛn.", "twi", codeswitch=True)
        self.assertIn("haɪ skuːl", ipa)
        self.assertIn("ɹiːd͡ʒənəl hɑːspɪɾəl", ipa)


if __name__ == "__main__":
    unittest.main()
