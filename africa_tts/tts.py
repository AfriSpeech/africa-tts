"""AfricaTTS: load a model, convert text to IPA, synthesise with a default or custom voice.

The model (about 440 MB) is downloaded from Hugging Face once, after asking, and cached; later
runs work offline. Skip the question with download=True (Python), --yes (CLI) or
AFRICA_TTS_DOWNLOAD=1.
"""

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from pocket_tts import TTSModel

from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL as _EXPERIMENTAL
from africa_tts.languages import LANGS as _LANGS
from africa_tts.languages import experimental_languages, languages  # noqa: F401 (re-export)

MODELS = {
    "africa-tts": "AfriSpeech/africa-tts",
    "africa-tts-turbo": "AfriSpeech/africa-tts",  # former name
}
DOWNLOAD_MB = 440
_MODEL_FILES = ["config.yaml", "model.safetensors", "tokenizer.json"]


class DownloadDeclined(RuntimeError):
    """The model is not downloaded yet and the download was not approved."""


def _interactive() -> bool:
    return bool(sys.stdin and sys.stdin.isatty()) or "ipykernel" in sys.modules


def ensure_downloaded(repo: str, download: bool | None = None) -> None:
    """Download `repo`'s model files and voices once, after approval; no-op when cached.

    `download`: None asks first in a terminal or notebook (elsewhere it raises DownloadDeclined
    rather than downloading unasked); True downloads without asking; False never downloads.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns = _MODEL_FILES + ["voices/*.wav"]
    try:  # already downloaded: no question, no network
        cached = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
        if all((cached / f).exists() for f in _MODEL_FILES):
            return
    except LocalEntryNotFoundError:
        pass
    if download is None and os.environ.get("AFRICA_TTS_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        download = True
    if download is None:
        if not _interactive():
            raise DownloadDeclined(
                f"Africa TTS is not downloaded yet ({repo}, about {DOWNLOAD_MB} MB). Approve the "
                "download with AfricaTTS.load(download=True), `africa-tts --download`, or "
                "AFRICA_TTS_DOWNLOAD=1.")
        answer = input(f"Download Africa TTS ({repo}, about {DOWNLOAD_MB} MB) from Hugging Face? "
                       "It is cached for later runs. [Y/n] ")
        download = answer.strip().lower() in ("", "y", "yes")
    if not download:
        raise DownloadDeclined("download declined; nothing was downloaded")
    snapshot_download(repo, allow_patterns=patterns)


class AfricaTTS:
    """Africa TTS with the africa-g2p front end and a default voice per language.

        tts = AfricaTTS.load()
        audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")   # float32, 24 kHz
        tts.save(audio, "out.wav")
    """

    def __init__(self, model: TTSModel, repo: str) -> None:
        self.model = model
        self.repo = repo
        self.sample_rate = model.sample_rate

    @classmethod
    def load(cls, name: str = "africa-tts", download: bool | None = None, device: str = "cpu",
             **kwargs) -> "AfricaTTS":
        """Load Africa TTS. The first time, it is downloaded from Hugging Face (about 440 MB) and
        cached. `download`: None (default) asks first in a terminal or notebook (elsewhere it
        raises DownloadDeclined instead of downloading unasked); True downloads without asking;
        False never downloads. `device`: "cpu" (default) or "cuda" for an NVIDIA GPU (needs the
        CUDA build of PyTorch). `name` can also be a local folder with config.yaml. Extra keyword
        arguments go to pocket_tts.TTSModel.load_model (e.g. temp)."""
        repo = MODELS.get(name, name)
        local = Path(repo)
        if local.is_dir():
            config = str(local / "config.yaml")
        else:
            ensure_downloaded(repo, download)
            config = f"hf://{repo}/config.yaml"
        model = TTSModel.load_model(config=config, **kwargs)
        if device != "cpu":
            model = model.to(device)
        return cls(model, repo)

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
