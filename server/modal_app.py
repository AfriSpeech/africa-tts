"""Deploy the Africa TTS API server (server/app.py) on Modal, CPU only.

    modal deploy server/modal_app.py          # from the repository root

Same server as the Docker image: sherpa-onnx + africa-g2p + espeak-ng 1.51 (Debian bookworm), with
the Turbo sherpa-onnx bundle baked into the image so a cold container starts in seconds.
Containers scale to zero when idle and up to MAX_CONTAINERS under load; each runs WORKERS model
instances (WORKERS requests in parallel).

It is deployed as a public demo API (it powers the AfriSpeech/africa-tts-demo Space): anonymous
callers are rate limited per IP with a daily budget (server/limits.py), text is capped, and at most
MAX_CONTAINERS run, which bounds the cost. Nothing runs when idle (scale to zero after 5 minutes).
Callers with the API key are not limited: put it in a Modal secret named "africa-tts-api-key"
(modal secret create africa-tts-api-key AFRICA_TTS_API_KEY=...); never in a web page.

Settings (environment at deploy time):
    AFRICA_TTS_MODEL         HF repo of the bundle (default AfriSpeech/africa-tts-turbo)
    AFRICA_TTS_CORS_ORIGINS  browser origins allowed (default: the demo Space)
    AFRICA_TTS_MAX_CHARS     text limit per request (default 500)
    AFRICA_TTS_RATE_PER_MIN, AFRICA_TTS_CHARS_PER_HOUR, AFRICA_TTS_CHARS_PER_DAY   see limits.py
    AFRICA_TTS_MAX_CONTAINERS / AFRICA_TTS_MIN_CONTAINERS   scaling bounds (default 2 / 0)
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("AFRICA_TTS_MODEL", "AfriSpeech/africa-tts-turbo")
WORKERS, THREADS = 2, 2
SPACE_ORIGIN = "https://afrispeech-africa-tts-demo.static.hf.space"


def _download_bundle() -> None:
    from huggingface_hub import snapshot_download

    root = snapshot_download(MODEL, allow_patterns=["sherpa-onnx/*"])
    os.symlink(Path(root) / "sherpa-onnx", "/bundle")


image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("espeak-ng", "git")  # espeak-ng 1.51, as in training
    .pip_install_from_requirements(str(ROOT / "server" / "requirements.txt"))
    .env({"AFRICA_TTS_MODEL": MODEL})
    .run_function(_download_bundle)
    .env({
        "AFRICA_TTS_BUNDLE": "/bundle",
        "AFRICA_TTS_WORKERS": str(WORKERS),
        "AFRICA_TTS_THREADS": str(THREADS),
        "AFRICA_TTS_MAX_CHARS": os.environ.get("AFRICA_TTS_MAX_CHARS", "500"),
        "AFRICA_TTS_CORS_ORIGINS": os.environ.get("AFRICA_TTS_CORS_ORIGINS", SPACE_ORIGIN),
        "AFRICA_TTS_PUBLIC": "1",
        "AFRICA_TTS_PROXY_HOPS": "0",  # Modal passes the real client IP as the peer
        "AFRICA_TTS_RATE_PER_MIN": os.environ.get("AFRICA_TTS_RATE_PER_MIN", "6"),
        "AFRICA_TTS_CHARS_PER_HOUR": os.environ.get("AFRICA_TTS_CHARS_PER_HOUR", "3000"),
        "AFRICA_TTS_CHARS_PER_DAY": os.environ.get("AFRICA_TTS_CHARS_PER_DAY", "200000"),
        "PYTHONPATH": "/root",
    })
    .add_local_dir(str(ROOT / "africa_tts"), "/root/africa_tts", ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "server"), "/root/server", ignore=["__pycache__"])
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
    from server.app import app as fastapi_app

    return fastapi_app
