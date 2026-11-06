# Africa TTS API server (CPU, sherpa-onnx). Build from the repository root:
#   docker build -f server/Dockerfile -t africa-tts-server .
#   docker run -p 8000:8000 -e AFRICA_TTS_WORKERS=4 -e AFRICA_TTS_THREADS=2 africa-tts-server
# The model bundle is downloaded from Hugging Face on first start (mount a volume on
# /root/.cache/huggingface to keep it), or bake it in with AFRICA_TTS_BUNDLE.
#
# espeak-ng (English) comes from Debian bookworm, version 1.51, the version the model was
# trained with (1.52 changes ~2% of English words). Its data is GPL-3.0-or-later.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY africa_tts africa_tts
COPY server server

ENV AFRICA_TTS_MODEL=AfriSpeech/africa-tts-turbo \
    AFRICA_TTS_WORKERS=2 \
    AFRICA_TTS_THREADS=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
