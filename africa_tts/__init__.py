"""Africa TTS: Pocket TTS for 92 African languages and English, reading IPA from africa-g2p.

The text front end (`to_ipa`) and the language lists load without PyTorch; `AfricaTTS` (and
pocket-tts / torch with it) is imported on first use, so a sherpa-onnx deployment such as the
API server in server/ needs neither.
"""

from africa_tts.frontend import to_ipa
from africa_tts.languages import experimental_languages, languages

__all__ = ["MODELS", "AfricaTTS", "experimental_languages", "languages", "to_ipa"]


def __getattr__(name: str):  # lazy: pocket-tts / torch only when the PyTorch path is used
    if name in ("AfricaTTS", "MODELS"):
        from africa_tts import tts

        return getattr(tts, name)
    raise AttributeError(name)
