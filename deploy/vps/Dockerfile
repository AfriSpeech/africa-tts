# Africa TTS API server for a VPS, PyTorch on CPU (fp32 unquantised quality).
# Every request needs the API key; the container refuses to start without AFRICA_TTS_API_KEY,
# since the server lets anyone in when no key is set.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

ARG AFRICA_TTS_VERSION=0.5.0
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "africa-tts[torch]==${AFRICA_TTS_VERSION}"

# Bake the model (about 440 MB) into the image, so restarts don't download it again.
RUN africa-tts --download

ENV AFRICA_TTS_DEVICE=cpu \
    AFRICA_TTS_WORKERS=3 \
    AFRICA_TTS_MAX_CHARS=1000 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["sh", "-c", "if [ -z \"$AFRICA_TTS_API_KEY\" ]; then echo 'Set AFRICA_TTS_API_KEY in .env first.' >&2; exit 1; fi; exec uvicorn africa_tts.server.app:app --host 0.0.0.0 --port 8000"]
