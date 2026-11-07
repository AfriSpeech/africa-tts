"""Synthesis engine for the API server: a pool of AfricaTTS instances + default voices.

Each instance synthesises one request at a time, so concurrency comes from the pool: with W
workers x T threads, W requests run in parallel. The backend is the package's: audio.cpp (CPU,
default) or PyTorch (CPU or GPU, `pip install "africa-tts[torch]"`).
"""

import asyncio
import hashlib
import io
import os
import queue
import tempfile
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL, LANGS
from africa_tts.tts import AfricaTTS, trim_silence


@dataclass
class EngineConfig:
    model: str = "africa-tts"  # model name, HF repo id or local folder
    backend: str = "audiocpp"  # audiocpp (CPU) | torch (CPU or GPU)
    device: str = "cpu"
    workers: int = 2  # parallel requests
    threads: int = 2  # CPU threads per worker (audio.cpp)
    temperature: float = 0.3
    max_chars: int = 2000

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            model=os.environ.get("AFRICA_TTS_MODEL", "africa-tts"),
            backend=os.environ.get("AFRICA_TTS_BACKEND", "audiocpp"),
            device=os.environ.get("AFRICA_TTS_DEVICE", "cpu"),
            workers=int(os.environ.get("AFRICA_TTS_WORKERS", "2")),
            threads=int(os.environ.get("AFRICA_TTS_THREADS", "2")),
            temperature=float(os.environ.get("AFRICA_TTS_TEMPERATURE", "0.3")),
            max_chars=int(os.environ.get("AFRICA_TTS_MAX_CHARS", "2000")),
        )


class RequestError(ValueError):
    """A client error (bad language, text too long, unreadable voice)."""


class Engine:
    def __init__(self, cfg: EngineConfig):
        if cfg.device != "cpu":
            cfg.backend = "torch"
        self.cfg = cfg
        # A server has no one to ask, so starting it approves the model download.
        first = AfricaTTS.load(cfg.model, backend=cfg.backend, device=cfg.device,
                               threads=cfg.threads, temperature=cfg.temperature, download=True)
        self._pool: asyncio.Queue[AfricaTTS] = asyncio.Queue()
        self._pool.put_nowait(first)
        for _ in range(cfg.workers - 1):
            self._pool.put_nowait(AfricaTTS.load(str(first.folder), backend=cfg.backend,
                                                 device=cfg.device, threads=cfg.threads,
                                                 temperature=cfg.temperature))
        self.default = first
        self.sample_rate = AfricaTTS.sample_rate
        # Warm every instance (graph allocation, voice encoding, the streaming session) so the
        # first requests do not pay for it.
        for tts in list(self._pool._queue):  # noqa: SLF001 - filled above, not yet shared
            tts.generate("Akwaaba.", lang="twi")
            for _ in tts.stream("Akwaaba.", lang="twi"):
                pass
        self._voice_dir = Path(tempfile.mkdtemp(prefix="africa_tts_voices_"))
        self.languages = {c: {"name": v["name"], "status": "trained"} for c, v in LANGS.items()}
        self.languages.update({c: {"name": v["name"], "status": "experimental",
                                   "voice_from": v["nearest"]} for c, v in EXPERIMENTAL.items()})

    # ------------------------------------------------------------------ inputs

    def default_voice(self, lang: str) -> str:
        return str(self.default.default_voice(lang))

    def custom_voice(self, wav_bytes: bytes) -> str:
        """Store an uploaded voice clip (cached by content) and return its path."""
        path = self._voice_dir / f"{hashlib.sha1(wav_bytes).hexdigest()}.wav"
        if not path.exists():
            try:
                a, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
            except Exception as e:
                raise RequestError(f"cannot read voice audio: {e}") from e
            a = a.mean(1)
            a = trim_silence(a, sr)
            if len(a) < sr * 1.0:
                raise RequestError("voice audio must be at least 1 s (3-10 s recommended)")
            sf.write(str(path), a[: sr * 10], sr)  # the model uses at most 10 s
        return str(path)

    def prepare(self, text: str, lang: str) -> str:
        if lang not in self.languages:
            raise RequestError(f"unknown language {lang!r}; GET /v1/languages")
        text = " ".join(text.split())
        if not text:
            raise RequestError("empty text")
        if len(text) > self.cfg.max_chars:
            raise RequestError(f"text longer than {self.cfg.max_chars} characters")
        return to_ipa(text, lang)

    # ------------------------------------------------------------------ synthesis

    async def synthesize(self, ipa: str, voice: str) -> np.ndarray:
        tts = await self._pool.get()
        try:
            return await asyncio.to_thread(tts.engine.generate, ipa, voice)
        finally:
            self._pool.put_nowait(tts)

    async def stream(self, ipa: str, voice: str,
                     cancelled: threading.Event) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks as the engine produces them."""
        tts = await self._pool.get()
        chunks: queue.Queue = queue.Queue()

        def run() -> None:
            try:
                for c in tts.engine.stream(ipa, voice):
                    chunks.put(c)
                    if cancelled.is_set():
                        break
            except Exception as e:  # noqa: BLE001 - forwarded to the consumer
                chunks.put(e)
            finally:
                chunks.put(None)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            while True:
                item = await asyncio.to_thread(chunks.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancelled.set()
            await asyncio.to_thread(worker.join)
            self._pool.put_nowait(tts)


def to_wav(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
