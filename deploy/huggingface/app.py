"""Africa TTS API on a free Hugging Face Space (Gradio SDK).

Free Spaces only run Gradio apps, so Gradio starts the server as usual and the
ordinary Africa TTS API is attached under /tts-api:

    GET  /tts-api/health
    GET  /tts-api/v1/languages
    POST /tts-api/v1/tts        (needs the key in the AFRICA_TTS_API_KEY Space secret)
"""

import os

try:
    # ZeroGPU Spaces expect this before PyTorch loads, and at least one GPU function.
    # Speech is made on the CPU; the function below is never called.
    import spaces

    @spaces.GPU
    def _unused_gpu_function():
        return None
except ImportError:  # not on a ZeroGPU Space
    pass

if not os.environ.get("AFRICA_TTS_API_KEY"):
    # Without a key the server would let anyone in.
    raise SystemExit("Add AFRICA_TTS_API_KEY as a Space secret first (Settings → Variables and secrets).")

os.environ.setdefault("AFRICA_TTS_DEVICE", "cpu")
# One request is spoken at a time; the rest queue.
os.environ.setdefault("AFRICA_TTS_WORKERS", "1")
os.environ.setdefault("AFRICA_TTS_MAX_CHARS", "1000")
# Gradio's Node front end (SSR) answers unknown paths with the web page, hiding the API.
os.environ["GRADIO_SSR_MODE"] = "false"

import asyncio  # noqa: E402

import gradio as gr  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from africa_tts.server.app import app as api  # noqa: E402

API_PATH = "/tts-api"


async def _load_model():
    # A mounted app's own startup never runs, so load the model (and download it) here.
    await api.router.lifespan_context(api).__aenter__()


asyncio.run(_load_model())

with gr.Blocks(title="Africa TTS API") as page:
    gr.Markdown(
        "# Africa TTS API\n"
        "A private text-to-speech API for one app, built on "
        "[Africa TTS](https://huggingface.co/AfriSpeech/africa-tts) by AfriSpeech "
        "(CC-BY-NC-4.0). There is nothing to try here: requests need a secret key.\n\n"
        f"Is it running? See [{API_PATH}/health]({API_PATH}/health)."
    )

if __name__ == "__main__":
    # Gradio picks the host and port the Space expects.
    server, _, _ = page.launch(server_name="0.0.0.0", prevent_thread_lock=True, ssr_mode=False)
    # First in line, so none of Gradio's own routes can answer for the API.
    server.router.routes.insert(0, Mount(API_PATH, app=api))
    print(f"Africa TTS API attached at {API_PATH}", flush=True)
    page.block_thread()
