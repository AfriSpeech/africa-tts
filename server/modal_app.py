"""Deploy the Africa TTS API server (africa_tts/server) on Modal, PyTorch CPU.

    modal deploy server/modal_app.py          # from the repository root

The image is server/Dockerfile (PyTorch CPU, unquantised fp32 quality, espeak-ng 1.51),
with the model baked in so a cold container starts in seconds.
Containers scale to zero when idle and up to MAX_CONTAINERS under load; each runs WORKERS model
instances (WORKERS requests in parallel).
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("AFRICA_TTS_MODEL", "AfriSpeech/africa-tts")
WORKERS = 2
SPACE_ORIGIN = "https://afrispeech-africa-tts-demo.static.hf.space"


def _download_model() -> None:
    from africa_tts.tts import model_dir

    model_dir(MODEL, download=True)


image = (
    modal.Image.from_dockerfile(str(ROOT / "server" / "Dockerfile"), context_dir=str(ROOT))
    .env({"AFRICA_TTS_MODEL": MODEL})
    .run_function(_download_model)
    .env({
        "AFRICA_TTS_DEVICE": "cpu",
        "AFRICA_TTS_WORKERS": str(WORKERS),
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
    cpu=4,
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
