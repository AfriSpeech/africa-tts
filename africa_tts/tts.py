"""AfricaTTS: text -> IPA -> speech with sherpa-onnx (ONNX Runtime, int8). No PyTorch.

The model (the int8 sherpa-onnx bundle, about 190 MB) is downloaded from Hugging Face once, after
asking, and cached; later runs work offline. Skip the question with download=True (Python),
--yes (CLI) or AFRICA_TTS_DOWNLOAD=1.
"""

import os
import re
import sys
import warnings
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL as _EXPERIMENTAL
from africa_tts.languages import voice_language

MODELS = {
    "africa-tts": "AfriSpeech/africa-tts",
    "africa-tts-turbo": "AfriSpeech/africa-tts",  # former name
}
SAMPLE_RATE = 24000
DOWNLOAD_MB = 190
_BUNDLE_FILES = ["lm_main.int8.onnx", "lm_flow.int8.onnx", "encoder.onnx", "decoder.int8.onnx",
                 "text_conditioner.onnx", "vocab.json", "token_scores.json"]


class DownloadDeclined(RuntimeError):
    """The model is not downloaded yet and the download was not approved."""


def _interactive() -> bool:
    return bool(sys.stdin and sys.stdin.isatty()) or "ipykernel" in sys.modules


def _approve(repo: str, download: bool | None) -> bool:
    if download is None and os.environ.get("AFRICA_TTS_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        download = True
    if download is not None:
        return download
    if not _interactive():
        raise DownloadDeclined(
            f"Africa TTS is not downloaded yet ({repo}, about {DOWNLOAD_MB} MB). Approve the "
            "download with AfricaTTS.load(download=True), `africa-tts --download`, or "
            "AFRICA_TTS_DOWNLOAD=1.")
    answer = input(f"Download Africa TTS ({repo}, about {DOWNLOAD_MB} MB) from Hugging Face? "
                   "It is cached for later runs. [Y/n] ")
    return answer.strip().lower() in ("", "y", "yes")


def _bundle_dir(name: str, download: bool | None = None) -> Path:
    """A local bundle folder, or the cached download of a Hugging Face repo's sherpa-onnx/
    (downloaded once, after approval: see _approve)."""
    local = Path(name)
    if local.is_dir():
        sub = local / "sherpa-onnx"
        return sub if (sub / "vocab.json").exists() else local
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo = MODELS.get(name, name)
    patterns = [f"sherpa-onnx/{f}" for f in _BUNDLE_FILES] + ["sherpa-onnx/voices/*.wav"]
    try:  # already downloaded: no question, no network
        cached = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
        if all((cached / "sherpa-onnx" / f).exists() for f in _BUNDLE_FILES):
            return cached / "sherpa-onnx"
    except LocalEntryNotFoundError:
        pass
    if not _approve(repo, download):
        raise DownloadDeclined("download declined; nothing was downloaded")
    return Path(snapshot_download(repo, allow_patterns=patterns)) / "sherpa-onnx"


def _read_voice(path: str | Path) -> tuple[np.ndarray, int]:
    if str(path).startswith("hf://"):  # hf://<org>/<repo>/<file>[@revision]
        from huggingface_hub import hf_hub_download

        m = re.match(r"hf://([^/]+/[^/@]+)/([^@]+)(?:@(.+))?$", str(path))
        if not m:
            raise ValueError(f"bad hf:// path {path!r}")
        path = hf_hub_download(m[1], m[2], revision=m[3])
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if len(audio) < sr:
        raise ValueError("voice audio must be at least 1 s (3-10 s of clean speech recommended)")
    return audio[: sr * 10], sr  # the model uses at most 10 s


class AfricaTTS:
    """Africa TTS with the africa-g2p front end and a default voice per language.

        tts = AfricaTTS.load()                                         # downloads once, then cached
        audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")   # float32, 24 kHz
        tts.save(audio, "out.wav")
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, bundle: Path, threads: int, temperature: float = 0.3) -> None:
        self.bundle = Path(bundle)
        self.temperature = temperature
        b = self.bundle
        pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
            lm_main=str(b / "lm_main.int8.onnx"), lm_flow=str(b / "lm_flow.int8.onnx"),
            encoder=str(b / "encoder.onnx"), decoder=str(b / "decoder.int8.onnx"),
            text_conditioner=str(b / "text_conditioner.onnx"),
            vocab_json=str(b / "vocab.json"), token_scores_json=str(b / "token_scores.json"))
        self._tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(pocket=pocket, num_threads=threads,
                                                    provider="cpu")))

    @classmethod
    def load(cls, name: str = "africa-tts", threads: int | None = None,
             temperature: float = 0.3, download: bool | None = None) -> "AfricaTTS":
        """Load Africa TTS. The first time, it is downloaded from Hugging Face (about 190 MB) and
        cached. `download`: None (default) asks first, in a terminal or notebook (elsewhere it
        raises DownloadDeclined instead of downloading unasked); True downloads without asking;
        False never downloads. `name` can also be a local folder holding the sherpa-onnx bundle.
        `threads`: CPU threads (default: up to 4)."""
        threads = threads or min(4, os.cpu_count() or 1)
        return cls(_bundle_dir(name, download), threads, temperature)

    def default_voice(self, lang: str) -> Path:
        """The bundled voice prompt for `lang`: 3.5-6 s of speech in that language (for an
        experimental language, the voice of its closest trained language)."""
        try:
            return self.bundle / "voices" / f"{voice_language(lang)}.wav"
        except ValueError:
            raise ValueError(f"unknown language {lang!r}; see africa_tts.languages() and "
                             "africa_tts.experimental_languages()") from None

    @lru_cache(maxsize=32)  # noqa: B019 - one model per process in practice
    def _voice(self, voice: str) -> tuple[np.ndarray, int]:
        return _read_voice(voice)

    def _prepare(self, text: str, lang: str, voice, ipa: bool):
        if lang in _EXPERIMENTAL:
            e = _EXPERIMENTAL[lang]
            warnings.warn(f"{e['name']} ({lang}) is experimental: not in training; IPA coverage "
                          f"{e['coverage']:.0%}, default voice from {e['nearest_name']}. Please "
                          "report how it sounds: https://github.com/AfriSpeech/africa-tts/issues",
                          stacklevel=3)
        audio, sr = self._voice(str(voice) if voice else str(self.default_voice(lang)))
        g = sherpa_onnx.GenerationConfig()
        g.reference_audio, g.reference_sample_rate = audio, sr
        g.num_steps = 1
        g.silence_scale = 1.0
        g.extra = {"temperature": str(self.temperature)}
        text = " ".join(text.split())
        return (text if ipa else to_ipa(text, lang)), g

    def generate(self, text: str, lang: str, voice: str | Path | None = None,
                 ipa: bool = False) -> np.ndarray:
        """Synthesise `text` (spelling, or IPA with ipa=True) in language `lang`.

        `voice`: path (or hf:// URL) of a WAV to clone, 3-10 s of clean speech; default: the
        language's bundled voice. Returns float32 samples at 24 kHz.
        """
        ipa_text, g = self._prepare(text, lang, voice, ipa)
        return np.asarray(self._tts.generate(ipa_text, g).samples, dtype=np.float32)

    def stream(self, text: str, lang: str, voice: str | Path | None = None,
               ipa: bool = False) -> Iterator[np.ndarray]:
        """Like generate(), but yields audio chunks (about one per sentence) as they are made."""
        import queue
        import threading

        ipa_text, g = self._prepare(text, lang, voice, ipa)
        chunks: queue.Queue = queue.Queue()

        def on_samples(samples, progress) -> int:
            chunks.put(np.asarray(samples, dtype=np.float32).copy())
            return 1

        def run() -> None:
            try:
                self._tts.generate(ipa_text, g, callback=on_samples)
            except Exception as e:  # noqa: BLE001 - re-raised in the consumer
                chunks.put(e)
            finally:
                chunks.put(None)

        threading.Thread(target=run, daemon=True).start()
        while (item := chunks.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def save(self, audio: np.ndarray, path: str | Path) -> None:
        """Write a 24 kHz WAV (16-bit PCM)."""
        sf.write(str(path), np.clip(audio, -1.0, 1.0), self.sample_rate, subtype="PCM_16")
