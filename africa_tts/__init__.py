"""Africa TTS: Pocket TTS for 92 African languages and English, reading IPA from africa-g2p."""

from africa_tts.frontend import to_ipa
from africa_tts.tts import MODELS, AfricaTTS, languages

__all__ = ["MODELS", "AfricaTTS", "languages", "to_ipa"]
