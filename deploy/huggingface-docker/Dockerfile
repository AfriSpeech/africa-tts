# Africa TTS API server for a Hugging Face Space (Docker SDK), PyTorch on CPU.
# Every request needs the API key: add AFRICA_TTS_API_KEY as a Space secret. The container
# refuses to start without it, since the server lets anyone in when no key is set.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

ARG AFRICA_TTS_VERSION=0.5.0
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "africa-tts[torch]==${AFRICA_TTS_VERSION}"

# Spaces run the container as user 1000, so the model must be downloaded as that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user

# Bake the model (about 440 MB) into the image, so restarts don't download it again.
RUN africa-tts --download

# The free Space has 2 vCPUs: one request at a time, the rest queue.
ENV AFRICA_TTS_DEVICE=cpu \
    AFRICA_TTS_WORKERS=1 \
    AFRICA_TTS_MAX_CHARS=1000 \
    PYTHONUNBUFFERED=1
# Spaces send traffic to port 7860.
EXPOSE 7860
CMD ["sh", "-c", "if [ -z \"$AFRICA_TTS_API_KEY\" ]; then echo 'Add AFRICA_TTS_API_KEY as a Space secret first.' >&2; exit 1; fi; exec uvicorn africa_tts.server.app:app --host 0.0.0.0 --port 7860"]
