# Africa TTS API server, default engine: audio.cpp (q8_0) on the CPU, no PyTorch.
# Build from the repository root:
#   docker build -f server/Dockerfile -t africa-tts-server .
#   docker run -p 8000:8000 -e AFRICA_TTS_WORKERS=4 -e AFRICA_TTS_THREADS=2 africa-tts-server
# The model (about 150 MB) is downloaded from Hugging Face on first start; mount a volume on
# /root/.cache/huggingface to keep it. PyTorch engine (CPU or GPU): server/Dockerfile.torch.
#
# espeak-ng (English) comes from Debian bookworm, version 1.51, the version the model was
# trained with (1.52 changes ~2% of English words). Its data is GPL-3.0-or-later.

# ---- build: audio.cpp + the africa-tts wheel from this repository
FROM python:3.11-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends git g++ make \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir build cmake ninja
WORKDIR /src
COPY . .
RUN python third_party/audiocpp/build.py --jobs 4 && python -m build --wheel -o /wheels

# ---- runtime
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

ENV AFRICA_TTS_BACKEND=audiocpp \
    AFRICA_TTS_WORKERS=2 \
    AFRICA_TTS_THREADS=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "africa_tts.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
