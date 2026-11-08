"""Synthesis engine for the API server: a pool of AfricaTTS instances + default voices.

Each instance synthesises one request at a time, so concurrency comes from the pool: with W
workers, W requests run in parallel. The engine is PyTorch on CPU (default) or GPU.
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

from africa_tts.chunking import chunk_paragraphs
from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL, LANGS
from africa_tts.tts import AfricaTTS, trim_silence


@dataclass
class EngineConfig:
    model: str = "africa-tts"  # model name, HF repo id or local folder
    device: str = "cpu"  # cpu (PyTorch unquantised fp32) | cuda (GPU)
    workers: int = 2  # parallel requests
    temperature: float = 0.3
    max_chars: int = 2000

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            model=os.environ.get("AFRICA_TTS_MODEL", "africa-tts"),
            device=os.environ.get("AFRICA_TTS_DEVICE", "cpu"),
            workers=int(os.environ.get("AFRICA_TTS_WORKERS", "2")),
            temperature=float(os.environ.get("AFRICA_TTS_TEMPERATURE", "0.3")),
            max_chars=int(os.environ.get("AFRICA_TTS_MAX_CHARS", "2000")),
        )


class RequestError(ValueError):
    """A client error (bad language, text too long, unreadable voice)."""


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        # A server has no one to ask, so starting it approves the model download.
        first = AfricaTTS.load(cfg.model, device=cfg.device, temperature=cfg.temperature,
                               download=True)
        self._pool: asyncio.Queue[AfricaTTS] = asyncio.Queue()
        self._pool.put_nowait(first)
        for _ in range(cfg.workers - 1):
            self._pool.put_nowait(AfricaTTS.load(str(first.folder), device=cfg.device,
                                                 temperature=cfg.temperature))
        self.default = first
        self.sample_rate = AfricaTTS.sample_rate
        # Warm every instance so the first requests do not pay for it.
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

    def prepare(self, text: str, lang: str) -> list[str]:
        if lang not in self.languages:
            raise RequestError(f"unknown language {lang!r}; GET /v1/languages")
        text = text.strip()
        if not text:
            raise RequestError("empty text")
        if len(text) > self.cfg.max_chars:
            raise RequestError(f"text longer than {self.cfg.max_chars} characters")
        return [to_ipa(p, lang) for p in chunk_paragraphs(text)]

    # ------------------------------------------------------------------ synthesis

    async def synthesize(self, ipa_paras: list[str], voice: str) -> np.ndarray:
        tts = await self._pool.get()
        try:
            pause = np.zeros(int(0.25 * tts.sample_rate), dtype=np.float32)
            audio_list = []
            for i, p_ipa in enumerate(ipa_paras):
                p_audio = await asyncio.to_thread(tts.engine.generate, p_ipa, voice)
                if len(p_audio) > 0:
                    audio_list.append(p_audio)
                    if i < len(ipa_paras) - 1:
                        audio_list.append(pause)
            if not audio_list:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(audio_list)
        finally:
            self._pool.put_nowait(tts)

    async def stream(self, ipa_paras: list[str], voice: str,
                     cancelled: threading.Event) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks as the engine produces them paragraph by paragraph."""
        tts = await self._pool.get()
        chunks: queue.Queue = queue.Queue()
        pause = np.zeros(int(0.25 * tts.sample_rate), dtype=np.float32)

        def run() -> None:
            try:
                for i, p_ipa in enumerate(ipa_paras):
                    for c in tts.engine.stream(p_ipa, voice):
                        chunks.put(c)
                        if cancelled.is_set():
                            break
                    if cancelled.is_set():
                        break
                    if i < len(ipa_paras) - 1:
                        chunks.put(pause)
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
