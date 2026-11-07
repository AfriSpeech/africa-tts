"""AfricaTTS: text -> IPA -> speech, with a default voice per language.

Two engines run the same model:

    pip install africa-tts            AfricaTTS.load()                  audio.cpp, q8_0 GGUF, CPU (default)
    pip install "africa-tts[torch]"   AfricaTTS.load(backend="torch")   PyTorch, fp32, CPU or GPU
                                      AfricaTTS.load(device="cuda")     (device="cuda" implies torch)

The model is downloaded from Hugging Face once, after asking (about 150 MB for audio.cpp, 440 MB
for PyTorch), and cached; later runs work offline. Skip the question with download=True (Python),
--yes (CLI) or AFRICA_TTS_DOWNLOAD=1.
"""

import os
import re
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL as _EXPERIMENTAL
from africa_tts.languages import experimental_languages, languages  # noqa: F401 (re-export)
from africa_tts.languages import voice_language

MODELS = {
    "africa-tts": "AfriSpeech/africa-tts",
    "africa-tts-turbo": "AfriSpeech/africa-tts",  # former name
}
SAMPLE_RATE = 24000
GGUF = "gguf/africa-tts-q8_0.gguf"
BACKENDS = ("audiocpp", "torch")
# backend -> (files to fetch from the model repo, download size in MB)
_FILES = {
    "audiocpp": ([GGUF, "voices/*.wav"], 150),
    "torch": (["config.yaml", "model.safetensors", "tokenizer.json", "voices/*.wav"], 440),
}


class DownloadDeclined(RuntimeError):
    """The model is not downloaded yet and the download was not approved."""


def _interactive() -> bool:
    return bool(sys.stdin and sys.stdin.isatty()) or "ipykernel" in sys.modules


def _approve(repo: str, mb: int, download: bool | None) -> bool:
    if download is None and os.environ.get("AFRICA_TTS_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        download = True
    if download is not None:
        return download
    if not _interactive():
        raise DownloadDeclined(
            f"Africa TTS is not downloaded yet ({repo}, about {mb} MB). Approve the download with "
            "AfricaTTS.load(download=True), `africa-tts --download`, or AFRICA_TTS_DOWNLOAD=1.")
    answer = input(f"Download Africa TTS ({repo}, about {mb} MB) from Hugging Face? "
                   "It is cached for later runs. [Y/n] ")
    return answer.strip().lower() in ("", "y", "yes")


def model_dir(name: str = "africa-tts", backend: str = "audiocpp",
              download: bool | None = None) -> Path:
    """A local folder, or the cached download of the model repo's files for `backend`
    (downloaded once, after approval: see AfricaTTS.load)."""
    local = Path(name)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns, mb = _FILES[backend]
    repo = MODELS.get(name, name)
    must = [p for p in patterns if "*" not in p]
    try:  # already downloaded: no question, no network
        cached = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
        if all((cached / p).exists() for p in must):
            return cached
    except LocalEntryNotFoundError:
        pass
    if not _approve(repo, mb, download):
        raise DownloadDeclined("download declined; nothing was downloaded")
    return Path(snapshot_download(repo, allow_patterns=patterns))


def _resolve_hf(path: str | Path) -> str:
    if str(path).startswith("hf://"):  # hf://<org>/<repo>/<file>[@revision]
        from huggingface_hub import hf_hub_download

        m = re.match(r"hf://([^/]+/[^/@]+)/([^@]+)(?:@(.+))?$", str(path))
        if not m:
            raise ValueError(f"bad hf:// path {path!r}")
        return hf_hub_download(m[1], m[2], revision=m[3])
    return str(path)


def trim_silence(audio: np.ndarray, sr: int, pad_sec: float = 0.18, fade_ms: float = 15.0,
                 min_sec: float = 1.2) -> np.ndarray:
    """Trim leading and trailing dead silence while preserving natural voice decays,
    breaths and sentence endings with `pad_sec` cushion and gentle edge fades."""
    frame = int(0.02 * sr)
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(1))
    threshold = max(rms.max() * 0.02, 0.002)
    voiced = np.where(rms > threshold)[0]
    if len(voiced) == 0:
        return audio
    pad_frames = int(pad_sec * sr)
    s = max(0, voiced[0] * frame - pad_frames)
    e = min(len(audio), (voiced[-1] + 1) * frame + pad_frames)
    min_frames = int(min_sec * sr)
    if (e - s) < min_frames:
        needed = min_frames - (e - s)
        e = min(len(audio), e + needed)
        if (e - s) < min_frames:
            s = max(0, e - min_frames)
    trimmed = audio[s:e].copy()
    fade_len = int((fade_ms / 1000.0) * sr)
    if len(trimmed) > 2 * fade_len:
        fade_curve = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        if s > 0:
            trimmed[:fade_len] *= fade_curve
        if e < len(audio):
            trimmed[-fade_len:] *= fade_curve[::-1]
    return trimmed


def _read_voice(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(_resolve_hf(path), dtype="float32", always_2d=True)
    audio = audio.mean(1)
    audio = trim_silence(audio, sr)
    if len(audio) < sr:
        raise ValueError("voice audio must be at least 1 s (3-10 s of clean speech recommended)")
    return audio[: sr * 10], sr


def _real_gguf(path: Path) -> Path:
    """A real file with a .gguf name. The Hugging Face cache stores files as symlinks to
    extension-less blobs, and audio.cpp picks the loader from the resolved file's extension."""
    if not path.is_symlink():
        return path
    blob = path.resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "africa_tts" / "gguf"
    target = cache / f"{blob.name}.gguf"
    if not target.exists() or target.stat().st_size != blob.stat().st_size:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(blob, tmp)  # no extra disk space
        except OSError:
            import shutil

            shutil.copyfile(blob, tmp)  # other file system: one copy, once
        tmp.replace(target)
    return target


class _AudioCppEngine:
    def __init__(self, folder: Path, threads: int, temperature: float) -> None:
        from africa_tts import _audiocpp

        self._engine = _audiocpp.Engine(_real_gguf(folder / GGUF), threads, temperature)
        self._voices: dict[str, tuple[np.ndarray, int]] = {}

    def _voice(self, voice: str) -> tuple[np.ndarray, int]:
        if voice not in self._voices:
            self._voices[voice] = _read_voice(voice)
        return self._voices[voice]

    def generate(self, ipa: str, voice: str) -> np.ndarray:
        return self._engine.generate(ipa, self._voice(voice))

    def stream(self, ipa: str, voice: str) -> Iterator[np.ndarray]:
        yield from self._engine.stream(ipa, self._voice(voice))


class _TorchEngine:
    def __init__(self, folder: Path, device: str, temperature: float) -> None:
        from pocket_tts import TTSModel
        import yaml

        cfg = yaml.safe_load((folder / "config.yaml").read_text())  # point at the local files
        cfg["weights_path"] = str(folder / "model.safetensors")
        cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(folder / "tokenizer.json")
        import tempfile

        fd, local_cfg = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        Path(local_cfg).write_text(yaml.safe_dump(cfg, allow_unicode=True))
        self.model = TTSModel.load_model(config=local_cfg, temp=temperature)
        os.unlink(local_cfg)
        if device != "cpu":
            self.model = self.model.to(device)
        self._states: dict = {}

    def _state(self, voice: str):
        if voice not in self._states:
            self._states[voice] = self.model.get_state_for_audio_prompt(_resolve_hf(voice))
        return self._states[voice]

    def generate(self, ipa: str, voice: str) -> np.ndarray:
        audio = self.model.generate_audio(self._state(voice), ipa)
        return audio.detach().cpu().numpy().astype(np.float32)

    def stream(self, ipa: str, voice: str) -> Iterator[np.ndarray]:
        for chunk in self.model.generate_audio_stream(self._state(voice), ipa):
            yield chunk.detach().cpu().numpy().astype(np.float32)


def _torch_missing() -> ImportError:
    return ImportError('the PyTorch engine needs the torch extra: pip install "africa-tts[torch]" '
                       "(on a CPU-only Linux machine add --extra-index-url "
                       "https://download.pytorch.org/whl/cpu)")


class AfricaTTS:
    """Africa TTS with the africa-g2p front end and a default voice per language.

        tts = AfricaTTS.load()                                         # asks once, then cached
        audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")   # float32, 24 kHz
        tts.save(audio, "out.wav")
        for chunk in tts.stream("Akwaaba! Yɛfrɛ me Kofi.", lang="twi"): ...
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, engine, folder: Path, backend: str, device: str) -> None:
        self.engine, self.folder, self.backend, self.device = engine, Path(folder), backend, device

    @classmethod
    def load(cls, name: str = "africa-tts", backend: str | None = None, device: str = "cpu",
             threads: int | None = None, temperature: float = 0.3,
             download: bool | None = None) -> "AfricaTTS":
        """Load Africa TTS.

        `backend`: "audiocpp" (default: audio.cpp, q8_0, CPU, no PyTorch) or "torch" (PyTorch,
        fp32; needs `pip install "africa-tts[torch]"`). `device`: "cpu", or "cuda" for a GPU
        (implies backend="torch"). `download`: the first time, the model is downloaded from
        Hugging Face and cached; None (default) asks first in a terminal or notebook and
        elsewhere raises DownloadDeclined instead of downloading unasked; True downloads without
        asking; False never downloads. `name` can also be a local folder (a copy of the model
        repo). `threads`: CPU threads for audio.cpp (default: up to 4)."""
        backend = backend or ("torch" if device != "cpu" else "audiocpp")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
        if backend == "audiocpp" and device != "cpu":
            raise ValueError('audio.cpp runs on the CPU here; for a GPU use backend="torch" '
                             '(pip install "africa-tts[torch]")')
        if backend == "torch":
            try:
                import pocket_tts  # noqa: F401  (fail before downloading anything)
            except ImportError:
                raise _torch_missing() from None
        else:
            from africa_tts import _audiocpp

            _audiocpp.lib()  # fail before downloading anything
        folder = model_dir(name, backend, download)
        if backend == "audiocpp":
            engine = _AudioCppEngine(folder, threads or min(4, os.cpu_count() or 1), temperature)
        else:
            engine = _TorchEngine(folder, device, temperature)
        return cls(engine, folder, backend, device)

    def default_voice(self, lang: str) -> Path:
        """The bundled voice prompt for `lang`: 3.5-6 s of speech in that language (a female voice
        from africa-female-speech-v2-best60 for the 17 languages it covers, an AfriSpeech Selector
        recording for the others, Ghanaian English for English). For an experimental language, the voice of its
        closest trained language."""
        try:
            return self.folder / "voices" / f"{voice_language(lang)}.wav"
        except ValueError:
            raise ValueError(f"unknown language {lang!r}; see africa_tts.languages() and "
                             "africa_tts.experimental_languages()") from None

    def _prepare(self, text: str, lang: str, voice, ipa: bool) -> tuple[str, str]:
        if lang in _EXPERIMENTAL:
            e = _EXPERIMENTAL[lang]
            warnings.warn(f"{e['name']} ({lang}) is experimental: not in training; IPA coverage "
                          f"{e['coverage']:.0%}, default voice from {e['nearest_name']}. Please "
                          "report how it sounds: https://github.com/AfriSpeech/africa-tts/issues",
                          stacklevel=3)
        text = " ".join(text.split())
        return (text if ipa else to_ipa(text, lang)), str(voice or self.default_voice(lang))

    def generate(self, text: str, lang: str, voice: str | Path | None = None,
                 ipa: bool = False) -> np.ndarray:
        """Synthesise `text` (spelling, or IPA with ipa=True) in language `lang`.

        `voice`: path (or hf:// URL) of a WAV to clone, 3-10 s of clean speech; default: the
        language's bundled voice. Returns float32 samples at 24 kHz.
        """
        return self.engine.generate(*self._prepare(text, lang, voice, ipa))

    def stream(self, text: str, lang: str, voice: str | Path | None = None,
               ipa: bool = False) -> Iterator[np.ndarray]:
        """Like generate(), but yields float32 audio chunks as they are made."""
        yield from self.engine.stream(*self._prepare(text, lang, voice, ipa))

    def save(self, audio: np.ndarray, path: str | Path) -> None:
        """Write a 24 kHz WAV (16-bit PCM)."""
        import soundfile as sf

        sf.write(str(path), np.clip(audio, -1.0, 1.0), self.sample_rate, subtype="PCM_16")
