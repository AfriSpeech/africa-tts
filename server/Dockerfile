# Africa TTS API server, PyTorch engine (fp32 unquantised quality): CPU by default.
# Build from the repository root:
#   docker build -f server/Dockerfile -t africa-tts-server .
#   docker run -p 8000:8000 -e AFRICA_TTS_WORKERS=2 africa-tts-server
FROM python:3.11-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# Cache bust for package rebuild
ARG CACHE_DATE=2026-11-08-v2
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" ".[torch]" && rm -rf /src

ENV AFRICA_TTS_DEVICE=cpu \
    AFRICA_TTS_WORKERS=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "africa_tts.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
