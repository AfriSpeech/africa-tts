"""Africa TTS API server: audio.cpp on CPU by default, or PyTorch (CPU/GPU) with the torch extra.

    africa-tts serve                                   # audio.cpp, port 8000
    africa-tts serve --backend torch [--device cuda]   # pip install "africa-tts[torch]"
    uvicorn africa_tts.server.app:app --host 0.0.0.0 --port 8000   # configured by the environment

Endpoints
    GET  /health                    liveness + model info
    GET  /v1/languages              trained and experimental languages
    POST /v1/tts                    JSON {text, lang, voice_wav_base64?, format: wav|pcm} -> audio
    POST /v1/audio/speech           OpenAI-compatible: {input, voice: <language code>, response_format}
    WS   /v1/tts/stream             send {"text", "lang", "voice_wav_base64"?}; receive a JSON header,
                                    binary PCM16 (24 kHz mono) chunks as they are generated, then
                                    {"event": "done"} (or {"event": "error"})

Configuration (environment)
    AFRICA_TTS_MODEL     model name, HF repo or local folder (default africa-tts); downloaded on start
    AFRICA_TTS_BACKEND   audiocpp (default, CPU) or torch (needs the torch extra)
    AFRICA_TTS_DEVICE    cpu (default) or cuda (implies torch)
    AFRICA_TTS_WORKERS   parallel requests (model instances), default 2
    AFRICA_TTS_THREADS   CPU threads per instance (audio.cpp), default 2
    AFRICA_TTS_MAX_CHARS max text length, default 2000
    AFRICA_TTS_MAX_QUEUE requests allowed to wait for a worker before 503, default 32
    AFRICA_TTS_API_KEY   if set, required as "Authorization: Bearer <key>"
    AFRICA_TTS_CORS_ORIGINS  comma-separated origins allowed to call from a browser ("*" for any)
    AFRICA_TTS_PUBLIC    1 = anonymous clients allowed under rate limits (limits.py);
                         requests with the API key are not limited
"""

import base64
import contextlib
import json
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from africa_tts.server.engine import Engine, EngineConfig, RequestError, to_pcm16, to_wav
from africa_tts.server.limits import LimitExceeded, Limiter, client_ip

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = EngineConfig.from_env()
    STATE["engine"] = Engine(cfg)
    STATE["model"] = cfg.model
    STATE["max_queue"] = int(os.environ.get("AFRICA_TTS_MAX_QUEUE", "32"))
    STATE["inflight"] = 0
    STATE["limiter"] = Limiter()
    yield


app = FastAPI(title="Africa TTS", version="0.4.2", lifespan=lifespan)
if os.environ.get("AFRICA_TTS_CORS_ORIGINS"):  # browser pages on other origins (e.g. a demo Space)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ["AFRICA_TTS_CORS_ORIGINS"].split(",")],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Audio-Seconds", "X-Compute-Seconds", "Retry-After"],
    )


def _auth(authorization: str | None) -> bool:
    """Check access; True if the caller is anonymous and must be rate limited."""
    key = os.environ.get("AFRICA_TTS_API_KEY")
    if key and authorization == f"Bearer {key}":
        return False
    if os.environ.get("AFRICA_TTS_PUBLIC") == "1":
        return True
    if key:
        raise HTTPException(401, "invalid or missing API key")
    return False


def _limit(limited: bool, headers, peer: str | None, text: str) -> None:
    if limited:
        STATE["limiter"].check(client_ip(headers, peer), len(" ".join(text.split())))


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


async def _synthesize(text: str, lang: str, voice_b64: str | None, fmt: str,
                      limited: bool, request: Request) -> Response:
    eng: Engine = STATE["engine"]
    with _Slot():
        try:
            ipa = eng.prepare(text, lang)
            voice = _voice(eng, lang, voice_b64)
        except RequestError as e:
            raise HTTPException(400, str(e)) from e
        _limit(limited, request.headers, request.client.host if request.client else None, text)
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
    return {"status": "ok", "model": STATE["model"], "backend": eng.cfg.backend,
            "device": eng.cfg.device, "workers": eng.cfg.workers,
            "threads_per_worker": eng.cfg.threads, "inflight": STATE["inflight"],
            "sample_rate": eng.sample_rate}


@app.get("/v1/languages")
async def list_languages():
    return STATE["engine"].languages


@app.post("/v1/tts")
async def tts(req: TTSRequest, request: Request, authorization: str | None = Header(None)):
    limited = _auth(authorization)
    return await _synthesize(req.text, req.lang, req.voice_wav_base64, req.format, limited, request)


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest, request: Request,
                        authorization: str | None = Header(None)):
    limited = _auth(authorization)
    return await _synthesize(req.input, req.voice, None, req.response_format, limited, request)


@app.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket):
    try:
        limited = _auth(ws.headers.get("authorization"))
    except HTTPException:
        await ws.close(code=4401)
        return
    await ws.accept()
    eng: Engine = STATE["engine"]
    try:
        msg = json.loads(await ws.receive_text())
        req = TTSRequest(**msg)
        ipa = eng.prepare(req.text, req.lang)
        voice = _voice(eng, req.lang, req.voice_wav_base64)
        _limit(limited, ws.headers, ws.client.host if ws.client else None, req.text)
    except LimitExceeded as e:
        await ws.send_json({"event": "error", "message": str(e), "retry_after": e.retry_after})
        await ws.close(code=1008)
        return
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
        with contextlib.suppress(Exception):  # the client may already be gone
            await ws.send_json({"event": "error", "message": str(e)})
            await ws.close()
    finally:
        STATE["inflight"] -= 1


@app.exception_handler(LimitExceeded)
async def _limit_exceeded(_, exc: LimitExceeded):
    return JSONResponse({"detail": str(exc)}, status_code=429,
                        headers={"Retry-After": str(exc.retry_after)})


@app.exception_handler(RequestError)
async def _request_error(_, exc: RequestError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
