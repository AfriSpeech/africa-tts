"""Deploy the Africa TTS API server (africa_tts/server) on Modal, CPU only.

    modal deploy server/modal_app.py          # from the repository root

The image is server/Dockerfile (audio.cpp built from this repository, q8_0 GGUF, africa-g2p,
espeak-ng 1.51), with the model baked in so a cold container starts in seconds.
Containers scale to zero when idle and up to MAX_CONTAINERS under load; each runs WORKERS model
instances (WORKERS requests in parallel).

It is deployed as a public demo API (it powers the AfriSpeech/africa-tts-demo Space): anonymous
callers are rate limited per IP with a daily budget (africa_tts/server/limits.py), text is capped, and at most
MAX_CONTAINERS run, which bounds the cost. Nothing runs when idle (scale to zero after 5 minutes).
Callers with the API key are not limited: put it in a Modal secret named "africa-tts-api-key"
(modal secret create africa-tts-api-key AFRICA_TTS_API_KEY=...); never in a web page.

Settings (environment at deploy time):
    AFRICA_TTS_MODEL         model repo (default AfriSpeech/africa-tts)
    AFRICA_TTS_CORS_ORIGINS  browser origins allowed (default: the demo Space)
    AFRICA_TTS_MAX_CHARS     text limit per request (default 500)
    AFRICA_TTS_RATE_PER_MIN, AFRICA_TTS_CHARS_PER_HOUR, AFRICA_TTS_CHARS_PER_DAY   see limits.py
    AFRICA_TTS_MAX_CONTAINERS / AFRICA_TTS_MIN_CONTAINERS   scaling bounds (default 2 / 0)
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("AFRICA_TTS_MODEL", "AfriSpeech/africa-tts")
WORKERS, THREADS = 2, 2
SPACE_ORIGIN = "https://afrispeech-africa-tts-demo.static.hf.space"


def _download_model() -> None:
    from africa_tts.tts import model_dir

    model_dir(MODEL, "audiocpp", download=True)


image = (
    modal.Image.from_dockerfile(str(ROOT / "server" / "Dockerfile"), context_dir=str(ROOT))
    .env({"AFRICA_TTS_MODEL": MODEL})
    .run_function(_download_model)
    .env({
        "AFRICA_TTS_BACKEND": "audiocpp",
        "AFRICA_TTS_WORKERS": str(WORKERS),
        "AFRICA_TTS_THREADS": str(THREADS),
        "AFRICA_TTS_MAX_CHARS": os.environ.get("AFRICA_TTS_MAX_CHARS", "500"),
        "AFRICA_TTS_CORS_ORIGINS": os.environ.get("AFRICA_TTS_CORS_ORIGINS", SPACE_ORIGIN),
        "AFRICA_TTS_PUBLIC": "1",
        "AFRICA_TTS_PROXY_HOPS": "0",  # Modal passes the real client IP as the peer
        "AFRICA_TTS_RATE_PER_MIN": os.environ.get("AFRICA_TTS_RATE_PER_MIN", "6"),
        "AFRICA_TTS_CHARS_PER_HOUR": os.environ.get("AFRICA_TTS_CHARS_PER_HOUR", "3000"),
        "AFRICA_TTS_CHARS_PER_DAY": os.environ.get("AFRICA_TTS_CHARS_PER_DAY", "200000"),
        "HF_HUB_OFFLINE": "1",  # the model is in the image
    })
)

app = modal.App("africa-tts-api", image=image)


@app.function(
    cpu=WORKERS * THREADS,
    memory=4096,
    min_containers=int(os.environ.get("AFRICA_TTS_MIN_CONTAINERS", "0")),
    max_containers=int(os.environ.get("AFRICA_TTS_MAX_CONTAINERS", "2")),
    secrets=[modal.Secret.from_name("africa-tts-api-key")],
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=16)  # requests beyond WORKERS queue inside the container
@modal.asgi_app()
def api():
    from africa_tts.server.app import app as fastapi_app

    return fastapi_app
