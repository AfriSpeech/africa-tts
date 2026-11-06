"""Africa TTS API server (sherpa-onnx, CPU).

    AFRICA_TTS_MODEL=AfriSpeech/africa-tts-turbo uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health                    liveness + model info
    GET  /v1/languages              trained and experimental languages
    POST /v1/tts                    JSON {text, lang, voice_wav_base64?, format: wav|pcm} -> audio
    POST /v1/audio/speech           OpenAI-compatible: {input, voice: <language code>, response_format}
    WS   /v1/tts/stream             send {"text", "lang", "voice_wav_base64"?}; receive a JSON header,
                                    binary PCM16 (24 kHz mono) chunks as they are generated, then
                                    {"event": "done"} (or {"event": "error"})

Configuration (environment)
    AFRICA_TTS_MODEL     HF repo whose sherpa-onnx/ bundle to serve (default AfriSpeech/africa-tts-turbo)
    AFRICA_TTS_BUNDLE    local bundle folder (skips the download)
    AFRICA_TTS_WORKERS   parallel requests (model instances), default 2
    AFRICA_TTS_THREADS   ONNX Runtime threads per instance, default 2
    AFRICA_TTS_MAX_CHARS max text length, default 2000
    AFRICA_TTS_MAX_QUEUE requests allowed to wait for a worker before 503, default 32
    AFRICA_TTS_API_KEY   if set, required as "Authorization: Bearer <key>"
"""

import base64
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from server.engine import Engine, EngineConfig, RequestError, to_pcm16, to_wav

STATE: dict = {}


def _bundle_dir() -> Path:
    if os.environ.get("AFRICA_TTS_BUNDLE"):
        return Path(os.environ["AFRICA_TTS_BUNDLE"])
    from huggingface_hub import snapshot_download

    repo = os.environ.get("AFRICA_TTS_MODEL", "AfriSpeech/africa-tts-turbo")
    root = snapshot_download(repo, allow_patterns=["sherpa-onnx/*"])
    return Path(root) / "sherpa-onnx"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = EngineConfig.from_env()
    cfg.bundle = _bundle_dir()
    STATE["engine"] = Engine(cfg)
    STATE["model"] = os.environ.get("AFRICA_TTS_MODEL", str(cfg.bundle))
    STATE["max_queue"] = int(os.environ.get("AFRICA_TTS_MAX_QUEUE", "32"))
    STATE["inflight"] = 0
    yield


app = FastAPI(title="Africa TTS", version="0.1.0", lifespan=lifespan)


def _auth(authorization: str | None) -> None:
    key = os.environ.get("AFRICA_TTS_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise HTTPException(401, "invalid or missing API key")


class _Slot:
    """Admission control: reject with 503 when too many requests are already waiting."""

    def __enter__(self):
        eng: Engine = STATE["engine"]
        if STATE["inflight"] >= eng.cfg.workers + STATE["max_queue"]:
            raise HTTPException(503, "server busy, retry later")
        STATE["inflight"] += 1

    def __exit__(self, *exc):
        STATE["inflight"] -= 1


class TTSRequest(BaseModel):
    text: str
    lang: str = Field(description="ISO 639-3 code, e.g. twi, hau, yor, swh, eng")
    voice_wav_base64: str | None = Field(None, description="optional WAV (3-10 s) to clone")
    format: str = Field("wav", pattern="^(wav|pcm)$")


class SpeechRequest(BaseModel):  # OpenAI /v1/audio/speech shape
    input: str
    voice: str = Field(description="language code (its default voice), e.g. twi")
    model: str | None = None
    response_format: str = Field("wav", pattern="^(wav|pcm)$")
    speed: float | None = None  # accepted for compatibility, ignored


def _voice(eng: Engine, lang: str, b64: str | None):
    if not b64:
        return eng.default_voice(lang)
    try:
        return eng.custom_voice(base64.b64decode(b64, validate=True))
    except (ValueError, base64.binascii.Error) as e:
        raise RequestError(f"voice_wav_base64: {e}") from e


async def _synthesize(text: str, lang: str, voice_b64: str | None, fmt: str) -> Response:
    eng: Engine = STATE["engine"]
    with _Slot():
        try:
            ipa = eng.prepare(text, lang)
            voice = _voice(eng, lang, voice_b64)
        except RequestError as e:
            raise HTTPException(400, str(e)) from e
        t0 = time.monotonic()
        audio = await eng.synthesize(ipa, voice)
    secs = len(audio) / eng.sample_rate
    headers = {"X-Audio-Seconds": f"{secs:.2f}", "X-Compute-Seconds": f"{time.monotonic() - t0:.2f}"}
    if fmt == "pcm":
        return Response(to_pcm16(audio), media_type="audio/L16; rate=24000; channels=1", headers=headers)
    return Response(to_wav(audio, eng.sample_rate), media_type="audio/wav", headers=headers)


@app.get("/health")
async def health():
    eng: Engine = STATE["engine"]
    return {"status": "ok", "model": STATE["model"], "workers": eng.cfg.workers,
            "threads_per_worker": eng.cfg.threads, "inflight": STATE["inflight"],
            "sample_rate": eng.sample_rate}


@app.get("/v1/languages")
async def list_languages():
    return STATE["engine"].languages


@app.post("/v1/tts")
async def tts(req: TTSRequest, authorization: str | None = Header(None)):
    _auth(authorization)
    return await _synthesize(req.text, req.lang, req.voice_wav_base64, req.format)


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest, authorization: str | None = Header(None)):
    _auth(authorization)
    return await _synthesize(req.input, req.voice, None, req.response_format)


@app.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket):
    key = os.environ.get("AFRICA_TTS_API_KEY")
    if key and ws.headers.get("authorization") != f"Bearer {key}":
        await ws.close(code=4401)
        return
    await ws.accept()
    eng: Engine = STATE["engine"]
    try:
        msg = json.loads(await ws.receive_text())
        req = TTSRequest(**msg)
        ipa = eng.prepare(req.text, req.lang)
        voice = _voice(eng, req.lang, req.voice_wav_base64)
    except (RequestError, ValueError, TypeError) as e:
        await ws.send_json({"event": "error", "message": str(e)})
        await ws.close()
        return
    except WebSocketDisconnect:
        return
    if STATE["inflight"] >= eng.cfg.workers + STATE["max_queue"]:
        await ws.send_json({"event": "error", "message": "server busy, retry later"})
        await ws.close(code=1013)
        return
    STATE["inflight"] += 1
    cancelled = threading.Event()
    sent = 0
    try:
        await ws.send_json({"event": "start", "sample_rate": eng.sample_rate,
                            "encoding": "pcm_s16le", "channels": 1})
        async for chunk in eng.stream(ipa, voice, cancelled):
            await ws.send_bytes(to_pcm16(chunk))
            sent += len(chunk)
        await ws.send_json({"event": "done", "audio_seconds": round(sent / eng.sample_rate, 2)})
        await ws.close()
    except WebSocketDisconnect:
        cancelled.set()  # stop generating for a client that left
    except Exception as e:  # noqa: BLE001 - report and close
        cancelled.set()
        try:
            await ws.send_json({"event": "error", "message": str(e)})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
    finally:
        STATE["inflight"] -= 1


@app.exception_handler(RequestError)
async def _request_error(_, exc: RequestError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
