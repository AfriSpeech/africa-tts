"""Synthesis engine for the API server: a pool of sherpa-onnx Pocket TTS instances + the IPA
front end + default voices.

sherpa-onnx synthesises one request at a time per instance (no dynamic batching for Pocket
TTS), so concurrency comes from a pool of instances, each pinned to a few CPU threads: with
W workers x T threads, W requests run in parallel. Size W x T to the machine's cores.
"""

import asyncio
import hashlib
import io
import os
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from africa_tts.frontend import to_ipa
from africa_tts.languages import EXPERIMENTAL, LANGS, voice_language


@dataclass
class EngineConfig:
    bundle: Path  # unpacked sherpa-onnx bundle (lm_main.int8.onnx, voices/, g2p/, ...)
    workers: int = 2  # parallel requests
    threads: int = 2  # ONNX Runtime threads per worker
    temperature: float = 0.3
    num_steps: int = 1
    max_chars: int = 2000

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            bundle=Path(os.environ.get("AFRICA_TTS_BUNDLE", "bundle")),
            workers=int(os.environ.get("AFRICA_TTS_WORKERS", "2")),
            threads=int(os.environ.get("AFRICA_TTS_THREADS", "2")),
            temperature=float(os.environ.get("AFRICA_TTS_TEMPERATURE", "0.3")),
            max_chars=int(os.environ.get("AFRICA_TTS_MAX_CHARS", "2000")),
        )


class RequestError(ValueError):
    """A client error (bad language, text too long, unreadable voice)."""


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        b = cfg.bundle
        pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
            lm_main=str(b / "lm_main.int8.onnx"), lm_flow=str(b / "lm_flow.int8.onnx"),
            encoder=str(b / "encoder.onnx"), decoder=str(b / "decoder.int8.onnx"),
            text_conditioner=str(b / "text_conditioner.onnx"),
            vocab_json=str(b / "vocab.json"), token_scores_json=str(b / "token_scores.json"))
        model_cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=pocket, num_threads=cfg.threads, provider="cpu"))
        self._pool: asyncio.Queue[sherpa_onnx.OfflineTts] = asyncio.Queue()
        for _ in range(cfg.workers):
            self._pool.put_nowait(sherpa_onnx.OfflineTts(model_cfg))
        self.sample_rate = 24000
        self._voices: dict[str, tuple[np.ndarray, int]] = {}
        self.languages = {c: {"name": v["name"], "status": "trained"} for c, v in LANGS.items()}
        self.languages.update({c: {"name": v["name"], "status": "experimental",
                                   "voice_from": v["nearest"]} for c, v in EXPERIMENTAL.items()})

    # ------------------------------------------------------------------ inputs

    def default_voice(self, lang: str) -> tuple[np.ndarray, int]:
        code = voice_language(lang)
        if code not in self._voices:
            self._voices[code] = self._read_wav((self.cfg.bundle / "voices" / f"{code}.wav").read_bytes())
        return self._voices[code]

    def custom_voice(self, wav_bytes: bytes) -> tuple[np.ndarray, int]:
        key = hashlib.sha1(wav_bytes).hexdigest()
        if key not in self._voices:
            if len(self._voices) > 256:  # bound the cache of uploaded voices
                self._voices = {k: v for k, v in self._voices.items() if len(k) < 40}
            self._voices[key] = self._read_wav(wav_bytes)
        return self._voices[key]

    @staticmethod
    def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
        try:
            a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception as e:
            raise RequestError(f"cannot read voice audio: {e}") from e
        a = a.mean(1)
        if len(a) < sr * 1.0:
            raise RequestError("voice audio must be at least 1 s (3-10 s recommended)")
        return a[: sr * 10], sr  # sherpa-onnx uses at most 10 s

    def prepare(self, text: str, lang: str) -> str:
        if lang not in self.languages:
            raise RequestError(f"unknown language {lang!r}; GET /v1/languages")
        text = " ".join(text.split())
        if not text:
            raise RequestError("empty text")
        if len(text) > self.cfg.max_chars:
            raise RequestError(f"text longer than {self.cfg.max_chars} characters")
        return to_ipa(text, lang)

    def _gen_config(self, voice: tuple[np.ndarray, int]) -> sherpa_onnx.GenerationConfig:
        g = sherpa_onnx.GenerationConfig()
        g.reference_audio, g.reference_sample_rate = voice
        g.num_steps = self.cfg.num_steps
        g.silence_scale = 1.0
        g.extra = {"temperature": str(self.cfg.temperature)}
        return g

    # ------------------------------------------------------------------ synthesis

    async def synthesize(self, ipa: str, voice: tuple[np.ndarray, int]) -> np.ndarray:
        tts = await self._pool.get()
        try:
            audio = await asyncio.to_thread(tts.generate, ipa, self._gen_config(voice))
        finally:
            self._pool.put_nowait(tts)
        return np.asarray(audio.samples, dtype=np.float32)

    async def stream(self, ipa: str, voice: tuple[np.ndarray, int],
                     cancelled: threading.Event) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks as sherpa-onnx decodes them (per sentence chunk / frames)."""
        tts = await self._pool.get()
        chunks: queue.Queue = queue.Queue()

        def on_samples(samples: np.ndarray, progress: float) -> int:
            chunks.put(np.asarray(samples, dtype=np.float32).copy())
            return 0 if cancelled.is_set() else 1  # 0 stops generation

        def run() -> None:
            try:
                tts.generate(ipa, self._gen_config(voice), callback=on_samples)
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
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
