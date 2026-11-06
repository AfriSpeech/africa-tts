"""AfricaTTS: load a model, convert text to IPA, synthesise with a default or custom voice."""

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from pocket_tts import TTSModel

from africa_tts.frontend import to_ipa
from africa_tts.guidance import GuidedTTSModel

MODELS = {
    "africa-tts": "AfriSpeech/africa-tts",  # 24 layers; guided (default) or unguided (cfg=1.0)
    "africa-tts-turbo": "AfriSpeech/africa-tts-turbo",  # 6 layers, guidance distilled in
}
# Classifier-free guidance per model: 2.0 for the 24-layer model (two passes per frame, the
# setting its release was listened to with); Turbo was distilled from that guided output, so it
# runs with none.
DEFAULT_CFG = {"AfriSpeech/africa-tts": 2.0, "AfriSpeech/africa-tts-turbo": 1.0}
_HERE = Path(__file__).parent
_LANGS = json.loads((_HERE / "languages.json").read_text(encoding="utf-8"))
_EXPERIMENTAL = json.loads((_HERE / "languages_experimental.json").read_text(encoding="utf-8"))


def languages() -> dict[str, dict]:
    """{code: {"name", "hours"}} for every language the models were trained on."""
    return dict(_LANGS)


def experimental_languages() -> dict[str, dict]:
    """Languages the models were NOT trained on but that africa-g2p has an IPA table for:
    {code: {"name", "country", "nearest" (closest trained language, whose voice is used by
    default), "coverage" (share of its phonemes seen in training), "tier" (A/B/C), ...}}.
    Quality is unknown: please report how they sound (see docs/EXPERIMENTAL_LANGUAGES.md)."""
    return dict(_EXPERIMENTAL)


class AfricaTTS:
    """Africa TTS with the africa-g2p front end and a default voice per language.

        tts = AfricaTTS.load("africa-tts-turbo")          # 6 layers, guidance distilled in
        tts = AfricaTTS.load("africa-tts")                # 24 layers, guidance 2.0 (2 passes/frame)
        tts = AfricaTTS.load("africa-tts", cfg=1.0)       # 24 layers, unguided (1 pass/frame)
        audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")   # float32, 24 kHz
        tts.save(audio, "out.wav")
    """

    def __init__(self, model: TTSModel, repo: str, cfg: float = 1.0) -> None:
        self.model = model
        self.repo = repo
        self.cfg = cfg
        self.sample_rate = model.sample_rate

    @classmethod
    def load(cls, name: str = "africa-tts-turbo", cfg: float | None = None,
             **kwargs) -> "AfricaTTS":
        """`name`: "africa-tts", "africa-tts-turbo", a Hugging Face repo id, or a local folder
        with config.yaml. `cfg`: classifier-free guidance (default 2.0 for africa-tts, 1.0 =
        none for Turbo and anything else). Extra keyword arguments go to
        pocket_tts.TTSModel.load_model."""
        repo = MODELS.get(name, name)
        local = Path(repo)
        config = str(local / "config.yaml") if local.is_dir() else f"hf://{repo}/config.yaml"
        if cfg is None:
            cfg = DEFAULT_CFG.get(repo, 1.0)
        if repo == MODELS["africa-tts-turbo"] and cfg != 1.0:
            warnings.warn("Turbo already has guidance 2.0 built in; extra guidance over-sharpens it",
                          stacklevel=2)
        model = TTSModel.load_model(config=config, **kwargs)
        if cfg != 1.0:
            model = GuidedTTSModel.wrap(model, cfg)
        return cls(model, repo, cfg)

    def default_voice(self, lang: str) -> str:
        """The bundled voice prompt for `lang`: 3.5-6 s of speech in that language (a female voice
        from africa-female-speech-v2-best60 for the 17 languages it covers, a JW recording for the
        others, Ghanaian English for English). For an experimental language, the voice of its
        closest trained language. Alternatives: hf://AfriSpeech/africa-tts/voices/jw/<lang>.wav."""
        if lang in _EXPERIMENTAL:
            lang = _EXPERIMENTAL[lang]["nearest"]
        if lang not in _LANGS:
            raise ValueError(f"unknown language {lang!r}; see africa_tts.languages() and "
                             "africa_tts.experimental_languages()")
        return f"hf://{MODELS['africa-tts']}/voices/{lang}.wav"

    @lru_cache(maxsize=32)  # noqa: B019 - one model per process in practice
    def _voice_state(self, voice: str) -> dict:
        return self.model.get_state_for_audio_prompt(voice)

    def generate(self, text: str, lang: str, voice: str | Path | None = None,
                 ipa: bool = False) -> np.ndarray:
        """Synthesise `text` (spelling, or IPA with ipa=True) in language `lang`.

        `voice`: path or hf:// URL of a WAV to clone (3-10 s of clean speech); default: the
        language's bundled voice. Returns float32 samples at self.sample_rate.
        """
        if lang in _EXPERIMENTAL:
            e = _EXPERIMENTAL[lang]
            warnings.warn(f"{e['name']} ({lang}) is experimental: not in training; IPA coverage "
                          f"{e['coverage']:.0%}, default voice from {e['nearest_name']}. Please "
                          "report how it sounds: https://github.com/AfriSpeech/africa-tts/issues",
                          stacklevel=2)
        state = self._voice_state(str(voice) if voice else self.default_voice(lang))
        audio = self.model.generate_audio(state, text if ipa else to_ipa(text, lang))
        return audio.detach().cpu().numpy().astype(np.float32)

    def save(self, audio: np.ndarray, path: str | Path) -> None:
        import scipy.io.wavfile

        scipy.io.wavfile.write(str(path), self.sample_rate, audio)
