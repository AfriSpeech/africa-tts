"""Africa TTS: fast text-to-speech for 92 African languages and English, on CPU (sherpa-onnx).

    from africa_tts import AfricaTTS
    tts = AfricaTTS.load()
    tts.save(tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi"), "out.wav")
"""

from africa_tts.frontend import to_ipa
from africa_tts.languages import experimental_languages, languages

__all__ = ["MODELS", "AfricaTTS", "experimental_languages", "languages", "to_ipa"]


def __getattr__(name: str):  # lazy: ONNX Runtime only when synthesising
    if name in ("AfricaTTS", "MODELS"):
        from africa_tts import tts

        return getattr(tts, name)
    raise AttributeError(name)
