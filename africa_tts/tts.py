"""AfricaTTS: text -> IPA -> speech, with a default voice per language.

Powered by PyTorch (fp32 unquantised) on CPU (default) or GPU:

    pip install africa-tts
    tts = AfricaTTS.load()                                        # PyTorch, CPU (default)
    tts = AfricaTTS.load(device="cuda")                           # PyTorch, NVIDIA GPU

The model is downloaded from Hugging Face once, after asking (about 440 MB),
and cached; later runs work offline. Skip the question with download=True (Python),
--yes (CLI) or AFRICA_TTS_DOWNLOAD=1.
"""

import os
import re
import sys
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml

from africa_tts.chunking import chunk_paragraphs
from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL as _EXPERIMENTAL
from africa_tts.languages import experimental_languages, languages  # noqa: F401 (re-export)
from africa_tts.languages import voice_language

MODELS = {
    "africa-tts": "AfriSpeech/africa-tts",
    "africa-tts-turbo": "AfriSpeech/africa-tts",  # former name
}
SAMPLE_RATE = 24000
FILES = ["config.yaml", "model.safetensors", "tokenizer.json", "voices/*.wav"]
DOWNLOAD_SIZE_MB = 440


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


def model_dir(name: str = "africa-tts", backend: str = "torch",
              download: bool | None = None) -> Path:
    """A local folder, or the cached download of the model repo's files
    (downloaded once, after approval: see AfricaTTS.load)."""
    local = Path(name)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns, mb = FILES, DOWNLOAD_SIZE_MB
    repo = MODELS.get(name, name)
    must = [p for p in patterns if "*" not in p]
    try:  # already downloaded: no question, no network
        cached = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
        if all((cached / p).exists() for p in must) and (cached / "voices").is_dir():
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
    resolved = _resolve_hf(path)
    try:
        import soundfile as sf

        audio, sr = sf.read(resolved, dtype="float32", always_2d=True)
        audio = audio.mean(1)
    except ImportError:
        import scipy.io.wavfile

        sr, raw = scipy.io.wavfile.read(resolved)
        audio = raw.mean(1) if raw.ndim > 1 else raw
        if audio.dtype != np.float32:
            max_v = float(np.iinfo(audio.dtype).max) if np.issubdtype(audio.dtype, np.integer) else 1.0
            audio = (audio / max_v).astype(np.float32)

    audio = trim_silence(audio, sr)
    if len(audio) < sr:
        raise ValueError("voice audio must be at least 1 s (3-10 s of clean speech recommended)")
    return audio[: sr * 10], sr


class _TorchEngine:
    def __init__(self, folder: Path, device: str, temperature: float) -> None:
        from pocket_tts import TTSModel

        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        cfg["weights_path"] = str(folder / "model.safetensors")
        cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(folder / "tokenizer.json")

        fd, local_cfg = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        Path(local_cfg).write_text(yaml.safe_dump(cfg, allow_unicode=True))
        self.model = TTSModel.load_model(config=local_cfg, temp=temperature)
        os.unlink(local_cfg)
        if device != "cpu":
            self.model = self.model.to(device)
        self._states: dict = {}

    def _state(self, voice: str | Path):
        v_key = str(voice)
        if v_key not in self._states:
            resolved = Path(_resolve_hf(voice))
            if resolved.exists():
                self._states[v_key] = self.model.get_state_for_audio_prompt(resolved)
            else:
                import torch

                audio, _ = _read_voice(voice)
                tensor = torch.from_numpy(audio[None]).float()
                self._states[v_key] = self.model.get_state_for_audio_prompt(tensor)
        return self._states[v_key]

    def generate(self, ipa: str, voice: str | Path) -> np.ndarray:
        audio = self.model.generate_audio(self._state(voice), ipa)
        return audio.detach().cpu().numpy().astype(np.float32)

    def stream(self, ipa: str, voice: str | Path) -> Iterator[np.ndarray]:
        for chunk in self.model.generate_audio_stream(self._state(voice), ipa):
            yield chunk.detach().cpu().numpy().astype(np.float32)


class AfricaTTS:
    """Africa TTS: fast, high-fidelity speech synthesis using PyTorch on CPU or GPU.

        tts = AfricaTTS.load()                                         # asks once, then cached
        audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")   # float32, 24 kHz
        tts.save(audio, "out.wav")
        for chunk in tts.stream("Akwaaba! Yɛfrɛ me Kofi.", lang="twi"): ...
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, engine: _TorchEngine, folder: Path, device: str) -> None:
        self.engine, self.folder, self.device = engine, Path(folder), device

    @classmethod
    def load(cls, name: str = "africa-tts", backend: str | None = None, device: str = "cpu",
             threads: int | None = None, temperature: float = 0.3,
             download: bool | None = None) -> "AfricaTTS":
        """Load Africa TTS.

        `device`: "cpu" (default: PyTorch unquantised fp32 on CPU) or "cuda" (NVIDIA GPU).
        `download`: the first time, the model is downloaded from Hugging Face and cached;
        None (default) asks first in a terminal or notebook and elsewhere raises DownloadDeclined
        instead of downloading unasked; True downloads without asking; False never downloads.
        `name` can also be a local folder (a copy of the model repo)."""
        try:
            import pocket_tts  # noqa: F401
        except ImportError:
            raise ImportError('pocket-tts is required: pip install africa-tts '
                              '(or add --extra-index-url https://download.pytorch.org/whl/cpu for CPU PyTorch)') from None

        folder = model_dir(name, download=download)
        engine = _TorchEngine(folder, device, temperature)
        return cls(engine, folder, device)

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

    def _warn_experimental(self, lang: str) -> None:
        if lang in _EXPERIMENTAL:
            e = _EXPERIMENTAL[lang]
            warnings.warn(f"{e['name']} ({lang}) is experimental: not in training; IPA coverage "
                          f"{e['coverage']:.0%}, default voice from {e['nearest_name']}. Please "
                          "report how it sounds: https://github.com/AfriSpeech/africa-tts/issues",
                          stacklevel=3)

    def generate(self, text: str, lang: str, voice: str | Path | None = None,
                 ipa: bool = False, codeswitch: bool = True) -> np.ndarray:
        """Synthesise `text` (spelling, or IPA with ipa=True) in language `lang`.
        Supports multi-paragraph articles and long text with natural speech flow and intonation.

        `voice`: path (or hf:// URL) of a WAV to clone, 3-10 s of clean speech; default: the
        language's bundled voice. Returns float32 samples at 24 kHz.
        `codeswitch`: if True (default), embedded English words/entities are spoken in English.
        """
        self._warn_experimental(lang)
        voice_path = str(voice or self.default_voice(lang))
        paras = chunk_paragraphs(text)
        if not paras:
            return np.zeros(0, dtype=np.float32)

        pause = np.zeros(int(0.25 * self.sample_rate), dtype=np.float32)
        audio_list = []
        for i, p in enumerate(paras):
            p_ipa = p if ipa else to_ipa(p, lang, codeswitch=codeswitch)
            p_audio = self.engine.generate(p_ipa, voice_path)
            if len(p_audio) > 0:
                audio_list.append(p_audio)
                if i < len(paras) - 1:
                    audio_list.append(pause)
        if not audio_list:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(audio_list)

    def stream(self, text: str, lang: str, voice: str | Path | None = None,
               ipa: bool = False, codeswitch: bool = True) -> Iterator[np.ndarray]:
        """Like generate(), but yields float32 audio chunks as each sentence/paragraph is made."""
        self._warn_experimental(lang)
        voice_path = str(voice or self.default_voice(lang))
        paras = chunk_paragraphs(text)
        pause = np.zeros(int(0.25 * self.sample_rate), dtype=np.float32)

        for i, p in enumerate(paras):
            p_ipa = p if ipa else to_ipa(p, lang, codeswitch=codeswitch)
            yield from self.engine.stream(p_ipa, voice_path)
            if i < len(paras) - 1:
                yield pause

    def save(self, audio: np.ndarray, path: str | Path) -> None:
        """Write a 24 kHz WAV (16-bit PCM)."""
        import soundfile as sf

        sf.write(str(path), np.clip(audio, -1.0, 1.0), self.sample_rate, subtype="PCM_16")
